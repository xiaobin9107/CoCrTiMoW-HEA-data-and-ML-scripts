#!/bin/bash
>rf_middle_out

max=560
min=540

for i in {1..10};do awk -v min=${min} -v max=${max} '$2<max&&$2>min{print NR,$1,$2}' ${i}_rf_tmp >>rf_middle_out;done
for i in {1..10};do awk -v min=${min} -v max=${max} '$2<max&&$2>min{print NR,$1,$2}' ${i}_linear_tmp >>rf_middle_out;done
#for i in {1..10};do awk -v min=${min} -v max=${max} '$2<max&&$2>min{print NR,$1,$2}' ${i}_rbf_tmp >>rf_middle_out;done

#awk '{print $1}' rf_middle_out | sort -n -k1  | uniq -c | sort -n -k1 -r | sed -n '1,20p' | awk '{print $1,$2}'

linesnum=($(awk '{print $1}' rf_middle_out | sort -n -k1  | uniq -c | sort -n -k1 -r | sed -n '1,20p' | awk '{print $2}'))
echo ${linesnum[@]}
