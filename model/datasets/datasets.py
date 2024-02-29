def create_dataset(cfg, split='train'):
    dataset = None
    data_loader = None
    if cfg.data.dataset == 'rcc_dataset_mimic':
        from datasets.mimic_diff_dataset import RCCDataset_mimic, RCCDataLoader
        dataset = RCCDataset_mimic(cfg, split)
        data_loader = RCCDataLoader(
            dataset,
            batch_size=dataset.batch_size,
            shuffle=True if split == 'train' else False,
            # sampler=sampler if split=='train' else None,
            num_workers=cfg.data.num_workers)
    else:
        raise Exception('Unknown dataset: %s' % cfg.data.dataset)
    
    return dataset, data_loader
