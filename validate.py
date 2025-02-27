from func.validate_func import valid_data
from utils.log_tool import *

if __name__ == '__main__':
    log(r'[INFO] 验证 ......')
    train_ok, val_ok, test_ok = valid_data()
    if not train_ok & val_ok & test_ok:
        log(f'[ERROR] get error from validate operation , please check it !')
        quit()

    log(r'[INFO] 验证通过 ......')