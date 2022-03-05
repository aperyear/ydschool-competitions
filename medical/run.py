from main import main


if __name__ == '__main__':
    for i in range(5):
        main(fold=i, model_name='densenetblur121d', lr=3e-4, batch=16, epoch=120, save_name='aug-RSDG')