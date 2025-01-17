

import torch
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from petastorm import TransformSpec

class PetastormTrainer:

    def __init__(self, train_data, transform_spec_fn, model, criterion, optimizer, batch_size, num_epochs, input_colname, output_colname):
        self.batch_size = batch_size
        self.train_data = train_data
        spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, "file:///dbfs/tmp/petastorm/cache")
        self.converter_train = make_spark_converter(self.data_train)
        self.transform_spec_fn = transform_spec_fn

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.input_colname = input_colname
        self.output_colname = output_colname

        

    def train_one_epoch(self, train_dataloader_iter, steps_per_epoch):
        epoch_loss_vec = []

        for step in range(steps_per_epoch):
            batch = next(train_dataloader_iter)
            X = torch.tensor(batch[self.input_colname], dtype=torch.float32)
            y = torch.tensor(batch[self.output_colname], dtype=torch.float32)
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            epoch_loss_vec.append(loss.item())
            optimizer.step()
            if step % 1000 == 0:
              print(f'Step {step} : Loss = {loss}')
        return epoch_loss_vec
        
    def train(self):

        self.num_epochs = num_epochs
        with self.converter_train.make_torch_dataloader(transform_spec=self.transform_spec, 
                                                        batch_size=self.batch_size) as train_dataloader:

        
        train_dataloader_iter = iter(train_dataloader)
        steps_per_epoch = len(self.converter_train)//self.batch_size
        print(f'steps_per_epoch / batch_count : {steps_per_epoch} and train set size : {len(self.converter_train)}')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'training on {device}')
        model.to(device)
        criterion.to(device)

        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, self.num_epochs))
            print('-' * 10)
            self.model.train()
            self.loss_vec = []
            epoch_loss_vec = self.train_one_epoch(train_dataloader_iter = train_dataloader_iter, steps_per_epoch = steps_per_epoch)
            self.loss_vec.extend(epoch_loss_vec)
