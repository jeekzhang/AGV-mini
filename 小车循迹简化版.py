import RPi.GPIO as GPIO
import time
import cv2
import numpy as np

def sign(x):
    if x>0:
        return 1.0
    else:
        return -1.0

EA, I2, I1, EB, I4, I3, LS, RS = (13, 19, 26, 16, 20, 21, 6, 12)

# 设置GPIO口为输出
FREQUENCY = 50
GPIO.setmode(GPIO.BCM)
GPIO.setup([EA, I2, I1, EB, I4, I3], GPIO.OUT)
GPIO.setup([LS, RS],GPIO.IN)
GPIO.output([EA, I2, EB, I3], GPIO.LOW)
GPIO.output([I1, I4], GPIO.HIGH)

# 设置PWM波,频率为50Hz
pwma = GPIO.PWM(EA, FREQUENCY)
pwmb = GPIO.PWM(EB, FREQUENCY)

# pwm波控制初始化
pwma.start(0)
pwmb.start(0)

# center定义
center_now = 320
# 打开摄像头，图像尺寸640*480（长*高），opencv存储值为480*640（行*列）
cap = cv2.VideoCapture(0)

target = 320

try:
    while (1):
        ret, frame = cap.read()
        # 转化为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 大津法二值化
        retval, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        # 腐蚀，白区域变小
        dst = cv2.erode(dst, None, iterations=6)

        # 单看第400行的像素值s
        color = dst[400]
        # 找到黑色的像素点个数
        black_count = np.sum(color == 0)
        # 找到黑色的像素点索引
        black_index = np.where(color == 0)

        # 防止black_count=0的报错
        if black_count == 0:
            black_count = 1

        # 找到白色像素的中心点位置
        center_now = (black_index[0][black_count - 1] + black_index[0][0]) / 2

        # 计算出center_now与标准中心点的偏移量
        direction = center_now - 320

        print(direction)
        # 停止
        if abs(direction) > 250:
            pwma.ChangeDutyCycle(0)
            pwmb.ChangeDutyCycle(0)

        error = center_now - target

        if abs(error) > 70:
            error=abs(error)

        elif error > 0:
            pwma.ChangeDutyCycle(22)
            pwmb.ChangeDutyCycle(24)

        # 左转
        elif error < 0:
            pwma.ChangeDutyCycle(25)
            pwmb.ChangeDutyCycle(19)

        if cv2.waitKey(1) & 0xFF == ord('q')
            break
        else:
            time.sleep(0.05)

except KeyboardInterrupt:
    cap.release()
    cv2.destroyAllWindows()
    pwma.stop()
    pwmb.stop()
    GPIO.cleanup()
