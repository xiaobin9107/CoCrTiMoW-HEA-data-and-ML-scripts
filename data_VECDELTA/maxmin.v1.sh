#!/bin/bash
#for i in {1..10};do awk -F "," '{print $2,$3}' ${i}_rf_test.csv | sort -n -k1 | sed -n '$p';done
#for i in {1..10};do awk -F "," '{print $2,$3}' ${i}_rf_test.csv | sort -n -k1 -r | sed -n '$p';done

for sys in rbf rf linear;do
echo ${sys}
for i in {1..10};do 
    sysindex=($(sed -n '2,$p' ${i}_${sys}_test.csv | awk -F "," '{print $2,$3}' | sort -n -k1 | sed -n '$p'))
    #echo ${sysindex[@]}
    INDEXCOMP=($(sed -n '2,$p' compernent-index.csv | awk -F "," -v INDEX=${sysindex[1]} '$1==INDEX{print $2,$3,$4,$5,$6}'))
    #echo ${sysindex[@]} ${INDEXCOMP[@]}
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" ${sysindex[@]} ${INDEXCOMP[@]}
done

echo

for i in {1..10};do 
    #sed -n '2,$p' ${i}_rbf_test.csv | awk -F "," '{print $2,$3}' | sort -n -k1 | sed -n '1p';
    sysindex=($(sed -n '2,$p' ${i}_${sys}_test.csv | awk -F "," '{print $2,$3}' | sort -n -k1 -r | sed -n '$p'))
    #echo ${sysindex[@]}                                                                                                                                                            
    INDEXCOMP=($(sed -n '2,$p' compernent-index.csv | awk -F "," -v INDEX=${sysindex[1]} '$1==INDEX{print $2,$3,$4,$5,$6}'))
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" ${sysindex[@]} ${INDEXCOMP[@]}
done
echo
done
