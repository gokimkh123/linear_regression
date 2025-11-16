import pandas as pd
from linear_regression import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


#### stochasitc_gd()
#### x_list : input, y_list : ground truth, model : linear regression model, alpha : learning rate
def stochasitc_gd(x_list, y_list, model, alpha):
    ####### Write your code here - start
    n = len(x_list)
    index = np.arange(n)
    np.random.shuffle(index)

    for i in index:
        x_i = x_list[i]
        y_i = y_list[i]

        MSE_w, MSE_b = model.gradient_of_SE(x_i, y_i)
        model.update_params(MSE_w, MSE_b, alpha)
    ####### Write your code here - end
    return


#### batch_gd()
#### x_list : input, y_list : ground truth, model : linear regression model, alpha : learning rate
def batch_gd(x_list, y_list, model, alpha):
    ####### Write your code here - start
    n = len(x_list)

    sum_MSE_w = 0.0
    sum_MSE_b = 0.0

    for i in range(n):
        x_i = x_list[i]
        y_i = y_list[i]

        MSE_w, MSE_b = model.gradient_of_SE(x_i, y_i)
        sum_MSE_w += MSE_w
        sum_MSE_b += MSE_b
    mean_MSE_w = sum_MSE_w / n
    mean_MSE_b = sum_MSE_b / n

    model.update_params(mean_MSE_w, mean_MSE_b, alpha)
    ####### Write your code here - end
    return


def main(gd_mode, alpha):
    max_epochs = 1000

    # CSV 파일 경로 설정
    tr_file_path = '../data/train.csv'
    val_file_path = '../data/val.csv'
    test_file_path = '../data/test.csv'

    # CSV 파일을 DataFrame로 로드
    df_train = pd.read_csv(tr_file_path)
    df_val = pd.read_csv(val_file_path)
    df_test = pd.read_csv(test_file_path)
    # DataFrame의 첫 5행 출력
    print(df_train.head())
    print(df_val.head())


    tr_income = df_train['income'].values
    tr_happiness = df_train['happiness'].values

    val_income = df_val['income'].values
    val_happiness = df_val['happiness'].values

    test_income = df_test['income'].values
    test_happiness = df_test['happiness'].values

    model = LinearRegression()
    num_tr_data = len(tr_income)
    num_val_data = len(val_income)
    num_test_data = len(test_income)

    # (Phase 6) 학습 전 초기 파라미터 저장
    initial_w = model.w
    initial_b = model.b

    # 그래프 생성
    plt.figure(figsize=(8, 6))
    plt.scatter(tr_income, tr_happiness, label='real', color='blue')
    prediction = model.predict(tr_income)
    plt.plot(tr_income, prediction, label='regression', color='red')

    # 그래프 제목 및 축 레이블 추가
    plt.title('income vs. happiness (Training)')
    plt.xlabel('happiness')
    plt.ylabel('income')
    plt.legend()
    # 그리드 추가
    plt.grid(True)
    # 그래프 출력
    plt.show()

    if gd_mode == 0: # stochastic GD
        print("[[[[[ train with stochastic GD]]]]]]")
    else:
        print("[[[[[ train with batch GD]]]]]]")

    for e in range(0, max_epochs):
        if gd_mode == 0: # stochastic GD
            stochasitc_gd(x_list=tr_income, y_list=tr_happiness, model=model, alpha=alpha)
        else: # batch GD
            batch_gd(x_list=tr_income, y_list=tr_happiness, model=model, alpha=alpha)

        ## mse calculation
        ## Calculate MSE for training data at the end of each epoch - begin
        t_pred = model.predict(tr_income)
        t_error = t_pred - tr_happiness
        MSE = np.mean(t_error ** 2)
        ## Calculate MSE for training data at the end of each epoch - end
        print("epoch: %d, MSE: %.3f" % (e, MSE))

    ## OK. Here, we finished the training - write code for evaluation - begin
    # (Phase 3) 학습 종료 후, Training data 그래프 출력
    plt.figure(figsize=(8, 6))
    plt.scatter(tr_income, tr_happiness, label='real', color='blue')
    prediction = model.predict(tr_income)
    plt.plot(tr_income, prediction, label='regression', color='red')
    plt.title('Income vs. happiness (After Training - Training data)')
    plt.xlabel('happiness')
    plt.ylabel('income')
    plt.legend()
    plt.grid(True)
    plt.show()

    # (Phase 4) Validation data 그래프 출력
    plt.figure(figsize=(8, 6))
    plt.scatter(val_income, val_happiness, label='real', color='green')
    prediction_val = model.predict(val_income)
    plt.plot(val_income, prediction_val, label='regression', color='red')
    plt.title('Income vs. happiness (After Training - Validation data)')
    plt.xlabel('happiness')
    plt.ylabel('income')
    plt.legend()
    plt.grid(True)
    plt.show()

    # (Phase 5) Test data 그래프 출력
    plt.figure(figsize=(8, 6))
    plt.scatter(test_income, test_happiness, label='real', color='orange')
    prediction_test = model.predict(test_income)
    plt.plot(test_income, prediction_test, label='regression', color='red')
    plt.title('Income vs. happiness (After Training - Test data)')
    plt.xlabel('happiness')
    plt.ylabel('income')
    plt.legend()
    plt.grid(True)
    plt.show()

    # (Phase 6) 최종 결과 출력

    # Validation MSE 계산
    val_pred = model.predict(val_income)
    val_errors = val_pred - val_happiness
    val_mse = np.mean(val_errors ** 2)

    # Test MSE 계산
    test_pred = model.predict(test_income)
    test_errors = test_pred - test_happiness
    test_mse = np.mean(test_errors ** 2)

    print("\n" + "=" * 20)
    print("FINAL RESULT")
    print("=" * 20)
    print(f"Model param (before train): w: {initial_w:.3f}, b: {initial_b:.3f}")
    print(f"Model param (after train): w: {model.w:.3f}, b: {model.b:.3f}")
    print(f"validation MSE: {val_mse:.5f}")
    print(f"test MSE: {test_mse:.5f}")

    ## OK. Here, we finished the training - write code for evaluation - end

if __name__ == "__main__":
    # 1. gd_mode 입력 받기
    while True:
        gd_mode = int(input("학습 방식을 선택하세요 (0: Stochastic GD, 1: Batch GD): "))
        try:
            if gd_mode == 0:
                print(">>> Stochastic GD (확률적 경사 하강법)을 시작합니다.")
            else:
                # 0 이외의 값은 Batch GD
                gd_mode = 1  # 1이 아닌 다른 값(예: 2, 3)을 넣어도 1로 처리
                print(">>> Batch GD (배치 경사 하강법)을 시작합니다.")

            break  # 성공적으로 입력받았으면 루프 탈출
        except ValueError:
            # 숫자가 아닌 값(예: 'a', 'b', 'batch')을 입력하면 오류 메시지 출력
            print("잘못된 입력입니다. 0 또는 1을 숫자로 입력해주세요.")


    ########## learning rate
    alpha = 0.01
    main(gd_mode=gd_mode, alpha=alpha)
