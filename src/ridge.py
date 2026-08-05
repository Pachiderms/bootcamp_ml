from src.linear_regression import MyLinearRegression as MyLR
from src.reg_linear_grad import vec_reg_linear_grad
from src.linear_loss_reg import reg_loss_
from src.decorators import check_type_and_shape_fit_method


class MyRidge(MyLR):
    """
    Description:
    My personnal ridge regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=1000, lambda_=0.5):
        super().__init__(thetas, alpha, max_iter)
        self.lambda_ = lambda_

    def get_params_(self):
        return {
            "thetas": self.thetas,
            "alpha": self.alpha,
            "max_iter": self.max_iter,
            "lambda_": self.lambda_,
        }

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        return self

    @check_type_and_shape_fit_method
    def fit_(self, x, y):
        self.costs.clear()

        new_theta = self.thetas.copy()
        for _ in range(self.max_iter):
            y_hat = super().predict_(x)
            current_loss = reg_loss_(y, y_hat, new_theta, self.lambda_)
            self.costs.append(current_loss)

            grad = vec_reg_linear_grad(y, x, new_theta, self.lambda_)
            new_theta = new_theta - self.alpha * grad

        self.thetas = new_theta
        return self

    def loss_(self, y, y_hat):
        return reg_loss_(y, y_hat, self.thetas, self.lambda_)
