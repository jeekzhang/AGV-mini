# coding:utf-8
import RPi.GPIO as GPIO
import time
import cv2
import numpy as np
#前期准备部分

#限制较大值
def sign(x):
    if x > 0:
        return 1.0
    else:
        return -1.0

# 定义引脚
EA, I2, I1, EB, I4, I3 = (13, 19, 26, 16, 20, 21)
FREQUENCY = 50

# 设置GPIO口
GPIO.setmode(GPIO.BCM)

# 设置GPIO口为输出
GPIO.setup([EA, I2, I1, EB, I4, I3], GPIO.OUT)
GPIO.output([EA, I2, EB, I3], GPIO.LOW)
GPIO.output([I1, I4], GPIO.HIGH)

pwma = GPIO.PWM(EA, FREQUENCY)
pwmb = GPIO.PWM(EB, FREQUENCY)

# pwm波初始化
pwma.start(0)
pwmb.start(0)

# center定义
center_now = 320

# 打开摄像头，图像尺寸640*480（长*高），opencv存储值为480*640（行*列）
cap = cv2.VideoCapture(0)

# PID 定义和初始化三个error和adjust数据
error = [0.0] * 3
adjust = [0.0] * 3

# PID 参数配置、目标值、左右轮基准占空比和占空比偏差范围（根据实际情况调整）
kp = 1.3
ki = 0.5
kd = 0.2
target = 320
lspeed = 60
rspeed = 60
control = 35
ret, frame = cap.read()

#前期准备完毕，节省时间
print("准备完毕！按下Enter启动！")
input()

try:
    while True:
        ret, frame = cap.read()
        # 转化为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 大津法二值化
        ret, dst = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        # 膨胀，白区域变大
        dst = cv2.dilate(dst, None, iterations=2)
        cv2.imshow("镜头画面", dst)

        # 单看第400行的像素值s
        color = dst[400]
        # 找到black色的像素点个数
        black_count = np.sum(color == 0)

        # 防止black_count=0的报错
        if black_count == 0:
            continue
        else:
            black_index = np.where(color == 0)
        # 找到黑色像素的中心点位置
        center_now = (black_index[0][black_count - 1] + black_index[0][0]) / 2

        # 计算出center_now与标准中心点的偏移量
        direction = center_now - 320

        print("偏差值：", direction)

        # 更新PID误差
        error[0] = error[1]
        error[1] = error[2]
        error[2] = center_now - target

        # 更新PID输出（增量式PID表达式）
        adjust[0] = adjust[1]
        adjust[1] = adjust[2]
        adjust[2] = adjust[1] \
                    + kp * (error[2] - error[1]) \
                    + ki * error[2] \
                    + kd * (error[2] - 2 * error[1] + error[0]);
        print(adjust[2])

        # 饱和输出限制在control绝对值之内
        if abs(adjust[2]) > control:
            adjust[2] = sign(adjust[2]) * control
        # print(adjust[2])

        # 执行PID

        # 右转
        if adjust[2] > 20:
            pwma.ChangeDutyCycle(rspeed - adjust[2])
            pwmb.ChangeDutyCycle(lspeed)

        # 左转
        elif adjust[2] < -20:
            pwma.ChangeDutyCycle(rspeed)
            pwmb.ChangeDutyCycle(lspeed + adjust[2])

        else:
            pwma.ChangeDutyCycle(rspeed)
            pwmb.ChangeDutyCycle(lspeed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("结束！")
    pass
# 释放清理

cap.release()
cv2.destroyAllWindows()
pwma.stop()
pwmb.stop()
GPIO.cleanup()