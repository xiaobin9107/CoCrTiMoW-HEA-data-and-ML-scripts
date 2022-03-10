#!/bin/bash
#for i in {1..10};do awk -F "," '{print $2,$3}' ${i}_rf_test.csv | sort -n -k1 | sed -n '$p';done
#for i in {1..10};do awk -F "," '{print $2,$3}' ${i}_rf_test.csv | sort -n -k1 -r | sed -n '$p';done
for i in {1..10};do 
    sysindex=($(sed -n '2,$p' ${i}_rf_predict.csv | awk -F "," '{print $1,$2}' | sort -n -k2 | sed -n '$p'))
    #echo ${sysindex[@]}
    INDEXCOMP=($(sed -n '2,$p' compernent-index.csv | awk -F "," -v INDEX=${sysindex[0]} '$1==INDEX{print $2,$3,$4,$5,$6}'))
    #echo ${sysindex[@]} ${INDEXCOMP[@]}
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" ${sysindex[@]} ${INDEXCOMP[@]}
done | sort -n -k1

echo

for i in {1..10};do 
    #sed -n '2,$p' ${i}_rbf_test.csv | awk -F "," '{print $2,$3}' | sort -n -k1 | sed -n '1p';
    sysindex=($(sed -n '2,$p' ${i}_rf_predict.csv | awk -F "," '{print $1,$2}' | sort -n -k2 -r | sed -n '$p'))
    #echo ${sysindex[@]}                                                                                                                                                            
    INDEXCOMP=($(sed -n '2,$p' compernent-index.csv | awk -F "," -v INDEX=${sysindex[0]} '$1==INDEX{print $2,$3,$4,$5,$6}'))
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" ${sysindex[@]} ${INDEXCOMP[@]}
done | sort -n -k1
