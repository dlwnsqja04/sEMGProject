from week2_data import prepare_dataset
from week2_plots import learning_plot, preparation_plots
from week2_train import train


def main() -> None:
    prepare_dataset()
    preparation_plots()
    train(epochs=5)
    learning_plot()
    print('2주차 완료: results/week2 폴더에서 결과를 확인하세요.')


if __name__ == '__main__':
    main()
