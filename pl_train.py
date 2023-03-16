import os.path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
import argparse
from torch.utils.data import DataLoader
from unetppp import UNet_Nested
from data import Dataset
import pytorch_lightning as pl
from pytorch_lightning import loggers


class pointclassifier(pl.LightningModule):
    def __init__(self, loader_len):
        super().__init__()
        self.classifier = UNet_Nested()
        self.lr = args.lr
        self.entropy_loss = torch.nn.CrossEntropyLoss()
        self.step_size = loader_len * args.epochs

    def forward(self, x):
        out = self.classifier(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), weight_decay=5e-4, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, max_lr=self.lr * 10, base_lr=self.lr / 10,
                                                      step_size_down=self.step_size, cycle_momentum=False)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        out = self.classifier(x)
        loss = self.entropy_loss(out, y)
        out = torch.argmax(out, dim=1)
        self.log('train.loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.classifier(x)
        loss = self.entropy_loss(out, y)
        out = torch.argmax(out, dim=1)
        out = out[0].cpu().detach().numpy()
        out *= 255
        if not os.path.exists('./Layer_Segmentations/'):
            os.makedirs('./Layer_Segmentations/')
        cv2.imwrite(f'./Layer_Segmentations/{batch_idx:04d}.png', out)
        self.log('valid.loss', loss)


parser = argparse.ArgumentParser(description='BuildingSeg')
parser.add_argument('--exp_name', type=str, default='test', metavar='N',
                    help='Name of the experiment')
parser.add_argument('-b', '--batch_size', type=int, default=4, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('-e', '--epochs', type=int, default=400, metavar='N',
                    help='number of episode to train ')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--seed', type=int, default=413, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--gpu_ids', type=str, default='[1]',
                    help='induct fix id to train')
parser.add_argument('--test', type=bool, default=False,
                    help='decide whether to test')
parser.add_argument('--lr_find', type=bool, default=True,
                    help='decide whether to find lr')
args = parser.parse_args()


def main():
    pl.seed_everything(args.seed)
    wandb_logger = loggers.WandbLogger(name=args.exp_name, project='BuildingSeg')

    train_transform = A.Compose([
        A.RandomResizedCrop(384, 384),
        A.Flip(p=0.5),
        A.Normalize(),
        ToTensorV2()
    ])
    test_transform = A.Compose([
        A.Resize(384, 384),
        A.Normalize(),
        ToTensorV2()
    ])
    train_loader = DataLoader(Dataset(mode='train', transform=train_transform), num_workers=16, pin_memory=True,
                              persistent_workers=True, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Dataset(mode='val', transform=test_transform), num_workers=16, pin_memory=True,
                            persistent_workers=True, batch_size=1, shuffle=False)

    model = pointclassifier(len(train_loader))

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='valid.loss', mode='min',
                                                       filename='{epoch}-{valid.loss:.3f}',
                                                       dirpath='./result/' + args.exp_name + '/',
                                                       auto_insert_metric_name=False)

    if args.lr_find:
        trainer = pl.Trainer(gpus=eval(args.gpu_ids))
        lr_finder = trainer.tuner.lr_find(model, train_loader, val_loader)
        print(lr_finder.suggestion())
        model.hparams.lr = lr_finder.suggestion()
    trainer = pl.Trainer(max_epochs=args.epochs,
                         devices=eval(args.gpu_ids),
                         accelerator='gpu',
                         callbacks=[checkpoint_callback],
                         fast_dev_run=args.test,
                         logger=wandb_logger,
                         log_every_n_steps=5,
                         )

    print('=========start training========')
    trainer.fit(model,
                train_loader,
                val_loader,
                )


if __name__ == '__main__':
    main()
