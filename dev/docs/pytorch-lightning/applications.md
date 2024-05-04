# 应用

## MNIST和FashionMNIST

这里使用PyTorch Lightning去优化[PyTorch版本的训练过程](/pytorch/applications)。

这里对数据的获取以及创建模型不进行修改，主要修改的在模型训练方面。 创建一个LightningModule让他去负责模型的训练，测试以及验证。

```python
# define the LightningModule
class LitLetNet5(L.LightningModule):
    def __init__(self, model,  lr, momentum, *args, **kwargs):
        super().__init__()
        # save hyperparameters to hparams
        self.save_hyperparameters(ignore=['model'])
        self.model = model

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, labels = batch
        outputs = self.model(x)
        loss = nn.functional.nll_loss(outputs, labels)

        predictions = outputs.argmax(dim=1)
        accuracy = (predictions == labels).type(torch.float).sum().item()

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        self.log("train_acc", accuracy/len(labels))
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, labels = batch
        outputs = self.model(x)
        val_loss = nn.functional.nll_loss(outputs, labels)

        predictions = outputs.argmax(dim=1)
        accuracy = (predictions == labels).type(torch.float).sum().item()

        self.log("val_loss", val_loss)
        self.log("val_acc", accuracy/len(labels))

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, labels = batch
        outputs = self.model(x)
        test_loss = nn.functional.nll_loss(outputs, labels)

        predictions = outputs.argmax(dim=1)
        accuracy = (predictions == labels).type(torch.float).sum().item()

        self.log("test_loss", test_loss)
        self.log("test_acc", accuracy/len(labels))

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=self.hparams.lr,
                                    momentum=self.hparams.momentum)
        return optimizer
```

在这里实现调用，需要注意的事项：
- 不需要自己手动将model移动到device, 只需要在L.Trainer里面调用`accelerator="gpu"` 会自动检测去使用cuda/mps/cpu


```python
# Create an instance of our network
model = LetNet5()

# init the autoencoder
autoLetNet5 = LitLetNet5(model, hparams.lr, hparams.momentum)

# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = L.Trainer(
    # default_root_dir="logs",
    limit_train_batches=hparams.batch_size,
    max_epochs=hparams.num_epochs,
    accelerator="gpu",
    devices="auto")

trainer.fit(model=autoLetNet5, 
            train_dataloaders=train_loader,
            val_dataloaders=val_loader)

# test the model
trainer.test(autoLetNet5, dataloaders=test_loader)
```

同时通过self.log方式记录的数据会自动保存到tensorboard中, 如果安装了tensorboard可以通过这个命令去查看数据 `tensorboard --logdir=lightning_logs/`, 这里lightning_logs是默认的数据存放路径，可以通过`default_root_dir`参数进行修改。
