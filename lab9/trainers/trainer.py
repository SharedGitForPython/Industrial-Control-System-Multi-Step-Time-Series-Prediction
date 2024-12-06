# trainers/trainer.py
import os
import numpy as np
import mindspore as ms
from mindspore import nn
import mindspore.ops as ops
import pandas


class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizer = nn.Adam(model.trainable_params(), config.learning_rate)
        self.loss_fn = nn.MAELoss(reduction='mean')
        def forward_fn(data, label):
            loss = 0
            x = data
            for step in range(config.horizon):
                pred = model(x, step)
                loss += self.loss_fn(pred, label[:,step:step+1,:])
                x = ops.concat((x[:,1:,:], pred), axis=1)
                x = x.copy()
                x[:,-1:,self.config.OP_index] = label[:,step:step+1,self.config.OP_index]
            return loss

        self.grad_fn = ms.value_and_grad(forward_fn, None, self.optimizer.parameters)
        self.patience = config.patience  # Add patience parameter to config
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

    def train_epoch(self, train_loader):
        self.model.set_train()
        epoch_loss = []

        for data, label in train_loader.create_tuple_iterator():
            loss, grads = self.grad_fn(data, label)
            self.optimizer(grads)
            epoch_loss.append(loss.asnumpy())
        succ_learn_rate = self.optimizer.learning_rate
        # if self.lr_scheduler:
        #     self.optimizer.learning_rate = self.lr_scheduler.get_lr()
        return np.mean(epoch_loss)

    def evaluate(self, data_loader):
        self.model.set_train(False)
        metrics = []

        for data, label in data_loader.create_tuple_iterator():
            x = data
            predictions = []
            
            for step in range(self.config.horizon):
                pred = self.model(x, step)  # [batch_size, 1, sensor_num]
                predictions.append(pred)
                x = ops.concat((x[:,1:,:], pred), axis=1)
                x = x.copy()
                x[:,-1:,self.config.OP_index] = label[:,step:step+1,self.config.OP_index]
            
            # Stack predictions along time dimension
            predictions = ops.stack(predictions, axis=1)  # [batch_size, horizon, sensor_num]
            
            # Calculate loss
            step_losses = []
            for step in range(self.config.horizon):
                step_loss = self.loss_fn(
                    predictions[:,step,:], 
                    label[:,step,:]
                )
                step_losses.append(step_loss.asnumpy())
            
            metrics.append(np.mean(step_losses))

        return np.mean(metrics)

    def train(self, train_loader, val_loader, test_loader):
        print("Starting training...")
        for epoch in range(self.config.num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.evaluate(val_loader)

            # Early stopping check
            if val_loss < self.best_val_loss*2.5:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                self.save_checkpoint('best_model.ckpt')
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
                
            print(f"Epoch {epoch+1}/{self.config.num_epochs}: "
                  f"Train Loss = {train_loss:.4f}, "
                  f"Val Loss = {val_loss:.4f}")
            print(f"Learning rate : {self.optimizer.learning_rate}")
        
        # Test
        print("\nEvaluating on test set...")
        self.load_checkpoint('best_model.ckpt')
        test_loss = self.evaluate(test_loader)
        print(f"Test Loss: {test_loss:.4f}")

    def save_checkpoint(self, filename):
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.config.checkpoint_dir, filename)
        ms.save_checkpoint(self.model, path)

    def load_checkpoint(self, filename):
        path = os.path.join(self.config.checkpoint_dir, filename)
        ms.load_checkpoint(path, net=self.model)