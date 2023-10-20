"""
其实对于批量归一化来讲，其实感觉就是控制输出的分布，将其限制为满足一定均值和方差的分布
使得其在训练时减少出现梯度消失和爆炸的问题，有点像正则化减少过拟合
值得注意的是批量归一化可以减少收敛的时间，加快收敛速度，但对于提高精度可能效果不是很明显。
"""