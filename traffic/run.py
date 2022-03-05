from main import main

if __name__ == '__main__':
    for i in range(5):
        main(fold=i, batch=32, week=6, save_name='week6')