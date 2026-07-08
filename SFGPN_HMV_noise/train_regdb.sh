#!/bin/bash

echo "开始执行训练脚本，trial参数从1到10"

for i in $(seq 1 10)
do
    echo "=== 第 $i 次执行，trial=$i ==="
    echo "开始时间: $(date)"

    # 执行训练命令
    python train.py --workers 8 --gpu 0 --dataset regdb --trial $i

    # 检查上一条命令的执行状态
    if [ $? -eq 0 ]; then
        echo "第 $i 次执行完成，trial=$i - 成功"
    else
        echo "第 $i 次执行完成，trial=$i - 失败"
    fi

    echo "结束时间: $(date)"
    echo "----------------------------------------"
done

echo "所有训练任务执行完毕！"