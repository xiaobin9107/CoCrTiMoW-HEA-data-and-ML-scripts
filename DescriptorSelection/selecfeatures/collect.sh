#!/bin/bash

#sed -n '2,$p' 1_rf_feature_importance.csv | awk -F "," '{print $1}' 

#sed -n '2,$p' 1_rf_feature_importance.csv | awk -F "," '{print $2}' | sort -n -r

cp ../*_rf_feature_importance.csv .

DESCR=($(sed -n '2,$p' 1_rf_feature_importance.csv | awk -F "," '{print $1}'))
#echo ${DESCR[@]}



for sys in *_rf_feature_importance.csv;do
    importancy=($(sed -n '2,$p' ${sys} | awk -F "," '{print $2}' | sort -n -r))
    importancy=($(sed -n '2,$p' ${sys} | awk -F "," '{print $1,$2}' | sort -n -k2 -r | awk '{print $2}'))
    descriptor=($(sed -n '2,$p' ${sys} | awk -F "," '{print $1,$2}' | sort -n -k2 -r | awk '{print $1}'))
    #echo
    #echo ${importancy[@]}
    importancy_max=${importancy[0]}
    importancy_min=${importancy[-1]}
    #echo ${importancy_max}
    #echo ${importancy_min}
    delta=$(echo ${importancy_max} ${importancy_min} | awk '{print $1-$2}')
    #echo ${delta}
    trans=()
    for((i=0;i<${#importancy[@]};i++));do
        trans_x=$(echo ${importancy[${i}]} | awk -v min=${importancy_min} -v del=${delta} '{printf"%.5f\n",($1-min)/del}')
        #echo ${descriptor[${i}]} ${importancy[${i}]} ${trans_x}
        echo " ${descriptor[${i}]}  ${trans_x}"
    done > ${sys}_trans
    #cat ${sys}_trans
done

for i in ${DESCR[@]};do
    #grep " ${i} " *_rf_feature_importance.csv_trans | awk '{print $3}'
    importancy_trans_arr=($(grep " ${i} " *_rf_feature_importance.csv_trans | awk '{print $3}'))
    
    importancy_trans_sum=0
    
    for j in ${importancy_trans_arr[@]};do
        importancy_trans_sum=$(echo ${importancy_trans_sum} ${j} | awk -v num=${#importancy_trans_arr[@]} '{print $1+$2/num}')
    done
    #echo ${importancy_trans_sum}    
    echo ${i} ${importancy_trans_sum} ${importancy_trans_arr[@]}  #| awk '{print NF}'
    #echo
done | sort -n -k2 -r | awk '$2>0.1{printf"\"%s\",",$1}'
echo 

rm *_rf_feature_importance.csv*
#rm *tmp
