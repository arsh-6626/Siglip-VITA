
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import copy
import gc
from transformers import SiglipModel, SiglipProcessor
from peft import get_peft_model, LoraConfig, TaskType
from DataLoder import VisionFineTuneDataset
from tqdm.auto import tqdm
from accelerate import Accelerator

def rescale_to_unit_range(tensor):
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    return torch.clamp(tensor, 0, 1)

def labels_to_prompt(image_path, label_dict, target_class=None):
    folder_map = {
        "trauma_head":"trauma_head",
        "trauma_torso":"trauma_torso",
        "trauma_lleg":"trauma_lower_ext",
        "trauma_rleg":"trauma_lower_ext",
        "trauma_larm":"trauma_upper_ext",
        "trauma_rarm":"trauma_upper_ext"
    }
    trauma_map = {
        "trauma_head": "head trauma",
        "trauma_torso": "torso trauma",
        "trauma_lower_ext": "lower extremity trauma",
        "trauma_upper_ext": "upper extremity trauma",
        "respiratory_distress": "respiratory distress",
        "severe_hemorrhage": "severe hemorrhage"
    }
    if target_class is None:
        return ""
    
    positive= []
    base = ""
    folder_name = os.path.basename(os.path.dirname(image_path))
    folder_class = folder_map.get(folder_name)
    if folder_class == target_class:
        for key, desc in trauma_map.items():
            if key in label_dict:
                if folder_class == key:
                    if label_dict[key] == "1":
                        positive.append(desc)
        if positive:
            base = "Body part shows " + ", ".join(positive)
    return base
    
def custom_collate_fn(batch):
    images, labels, paths = zip(*batch)
    return torch.stack(images, 0), list(labels), list(paths)

def get_lora_config():
    modules = []
    for i in range(26):
        modules += [
            f"vision_model.encoder.layers.{i}.self_attn.q_proj",
            f"vision_model.encoder.layers.{i}.self_attn.k_proj",
            f"vision_model.encoder.layers.{i}.self_attn.v_proj"
        ]
    return LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )

def siglip_contrastive_loss(image_embeds, text_embeds, temperature=0.07):
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    logits = (image_embeds @ text_embeds.T) / temperature
    N = logits.shape[0]
    labels = torch.eye(N, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, labels)

def similarity(image_embeds, text_embeds):
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    return image_embeds @ text_embeds.T


def train_class_adapter(class_name, dataset, epochs=20, batch_size=1, lr=5e-5):
    accelerator = Accelerator()
    # device = accelerator.device
    device = torch.device("cuda:0")
    print(f"training for class: {class_name}")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    print(f"Training - {len(train_dataset)}, val - {len(val_dataset)}")
    processor = SiglipProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    base_model = SiglipModel.from_pretrained("google/siglip-so400m-patch14-384")
    lora_config = get_lora_config()
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.to(device)
    # model, optimizer, train_loader, val_loader = accelerator.prepare(
    #     model, optimizer, train_loader, val_loader
    # )
    best_val_sim, best_model = -1.0, None
    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        total_loss, count = 0.0, 0
        for images, label_dicts, image_paths in tqdm(train_loader, desc="Training Batches"):
            images = [rescale_to_unit_range(img) for img in images]
            prompts = [labels_to_prompt(p, ld, target_class=class_name) for p, ld in zip(image_paths, label_dicts)]
            print(prompts)
            inputs = processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)
            outputs = model(**inputs)
            loss = siglip_contrastive_loss(outputs.image_embeds, outputs.text_embeds)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            count += 1
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/count:.4f}")
        model.eval()
        total_sim, pos_sim, neg_sim, p_count, n_count = 0, 0, 0, 0, 0
        for images, label_dicts in tqdm(val_loader, desc="Validation Batches"):
            prompts = [labels_to_prompt(ld, target_class=class_name) for ld in label_dicts]
            inputs = processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)
            outputs = model(**inputs)
            sims = similarity(outputs.image_embeds, outputs.text_embeds)
            diag = sims.diag().mean().item()
            total_sim += diag
            for i, ld in enumerate(label_dicts):
                if ld[class_name] == "1":
                    pos_sim += sims[i, i].item()
                    p_count += 1
                else:
                    neg_sim += sims[i, i].item()
                    n_count += 1
        avg_sim = total_sim / len(val_loader)
        if avg_sim > best_val_sim:
            best_val_sim = avg_sim
            best_model = copy.deepcopy(model)
            print(f"best model saved (sim: {best_val_sim:.4f})")
    print("Training complete")
    return best_model, processor, accelerator

if __name__ == "__main__":
    image_dir = './DTC-Trauma/main/frames'
    json_dir = './DTC-Trauma/main/bbox'
    output_dir = './lora_adapters_gauss_16'
    os.makedirs(output_dir, exist_ok=True)
    dataset = VisionFineTuneDataset(image_dir=image_dir, json_dir=json_dir)
    print(f"Dataset size: {len(dataset)} images")
    trauma_classes = [
        "trauma_head",
        "trauma_torso",
        "trauma_lower_ext",
        "trauma_upper_ext",
        "respiratory_distress",
        "severe_hemorrhage"
    ]
    for class_name in trauma_classes:
        class_save_path = os.path.join(output_dir, class_name)
        result = train_class_adapter(class_name, dataset)
        if result is not None:
            model, processor, accelerator = result
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                os.makedirs(class_save_path, exist_ok=True)
                unwrapped.save_pretrained(class_save_path)
                processor.save_pretrained(class_save_path)
                model.cpu()
                del model
                gc.collect()
                torch.cuda.empty_cache()

                print(f"Saved adapter for {class_name} to {class_save_path}")

    print("\nCompleted training all class-specific adapters!")
