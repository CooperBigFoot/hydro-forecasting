import pytorch_lightning as pl 


class HydroLazyDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule that lazily loads datasets.
    """

    def __init__(self, dataset_class, *args, **kwargs):
        super().__init__()
        self.dataset_class = dataset_class
        self.args = args
        self.kwargs = kwargs

    def prepare_data(self):
        # This method is called on every GPU
        # Use this to download data if needed
        pass

    def setup(self, stage=None):
        # This method is called on a single GPU
        # Use this to set up your datasets
        self.dataset = self.dataset_class(*self.args, **self.kwargs)