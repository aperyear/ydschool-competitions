from main import main
from default import config


if __name__ == '__main__':
    config['wandb_log'] = True
    config['epochs'] = 5
    config['target_col'] = 'bias'
    config['input_col'] = 'comment'
    config['weight_decay'] = 0.
    config['file_name'] = 'com_w0'
    main(**config, config=config)