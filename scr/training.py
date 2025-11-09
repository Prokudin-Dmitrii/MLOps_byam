import torch
from torch.optim import AdamW

from transformers import get_linear_schedule_with_warmup


def model_training(config_params, logger, model, tokenizer, train_dataloader, validation_dataloader=None):
    learning_rate = config_params['training']['learning_rate']
    weight_decay = config_params['training']['weight_decay']
    warmup_fraction = config_params['training']['warmup_fraction']
    n_steps = len(train_dataloader) * n_epochs
    
    n_epochs = config_params['training']['n_epochs']
    device = config_params['training']['device']

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = warmup_fraction * n_steps,
        num_training_steps = n_steps
    )

    scaler = torch.cuda.amp.GradScaler()

    few_shot_steps = config_params['training']['few_shot_steps']
    save_steps = config_params['training']['save_steps']
    logger_save_steps = config_params['training']['logger_save_steps']
    few_shots_examples = config_params['training']['few_shots_examples']
    few_shot_max_length = config_params['training']['few_shots_max_length']
    few_shot_sample = config_params['training']['few_shots_sample']
    few_shot_top_p = config_params['training']['few_shots_top_p']
    few_shot_temperature = config_params['training']['few_shot_temperature']
    few_shot_repertition_penalty = config_params['training']['few_shot_repertition_penalty']

    model_save_folder = config_params['training']['save_folder']
    if model_save_folder[-1] != '/':
        model_save_folder = model_save_folder + '/'

    step = 0
    losses = []

    model.to(device)

    logger.info(f'Старт обучения на устройстве {device}. Обучение длится {n_epochs} эпох, всего {n_steps} шагов.')
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        model.train()
        logger.info(f'Старт эпохи {epoch + 1} / {n_epochs}.')

        for _, batch in enumerate(train_dataloader):
            input_ids = batch[0]
            attention_masks = batch[1]
            labels = input_ids.clone()

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids.to(device), attention_mask=attention_masks.to(device), labels=labels.to(device))
                loss = outputs.loss

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            step += 1

            if step % logger_save_steps == 0:
                logger.info(f'Шаг {step} / {n_steps}, лосс на текущем шаге: {loss.item():.4f}')
            
            #writer.add_scalar('Train loss/step', loss.item(), step)
            #writer.add_scalar('LR/step', scheduler.get_last_lr()[0], step)
            
            if step % few_shot_steps == 0:
                model.eval()
                with torch.no_grad(), torch.cuda.amp.autocast():
                    for i, few_shots_example in enumerate(few_shots_examples):
                        ids = tokenizer(few_shots_example, return_tensors='pt', add_special_tokens=False).to(device)['input_ids']
                        outputs = model.generate(ids, max_length=few_shot_max_length, do_sample=few_shot_sample, top_p=few_shot_top_p, temperature=few_shot_temperature, repetition_penalty=few_shot_repertition_penalty)
                        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        
                        logger.info(f'Few-shot пример {i+1}: {result[:100]}...')
                        #writer.add_text(f'Few-show examples_{i}/step', result, step)

            if step % save_steps == 0:
                save_path = model_save_folder + f'model_checkpoint_{step}'
                model.save_pretrained(save_path)
                logger.info(f'Чекпоинт модели {step} шага обучения сохранён по пути: {save_path}')
        
        if validation_dataloader is not None:
            model.train()
            val_epoch_loss = 0
            for batch in validation_dataloader:
                with torch.no_grad():
                    input_ids = batch[0]
                    attention_masks = batch[1]
                    labels = input_ids.clone()

                    with torch.cuda.amp.autocast():
                        outputs = model(input_ids=input_ids.to(device), attention_mask=attention_masks.to(device), labels=labels.to(device))
                        loss = outputs.loss

                    val_epoch_loss += loss.item()
            
            mean_val_epoch_loss = val_epoch_loss / len(validation_dataloader)
            logger.info(f'Валидация эпохи {epoch + 1} / {n_epochs}, средний лосс на валидации: {mean_val_epoch_loss:.4f}')

        mean_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f'Конец эпохи {epoch + 1} / {n_epochs}, средний лосс за эпоху: {mean_epoch_loss:.4f}')
        #writer.add_scalar("Train epoch loss/epoch", mean_epoch_loss, epoch + 1)
        losses.append(mean_epoch_loss)

    logger.info(f'Конец обучения.')
    return model, losses