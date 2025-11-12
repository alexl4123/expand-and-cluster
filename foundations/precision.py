import torch

class Precision:
    @classmethod
    def get_precision_from_string(cls, precision):
        if precision is not None:
            if precision == "float32":
                precision = torch.float32
            elif precision == "float64":
                precision = torch.float64
            elif precision == "float128":
                precision = torch.float128
            else:
                print("NOT DEFINED PRECISION")
                print(precision)
                exit()
        else:
            precision = torch.float32

        return precision

