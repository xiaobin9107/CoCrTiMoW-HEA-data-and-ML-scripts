#!/bin/bash

[ $# -eq 0 ] && echo "Usage: $0 args" && exit

args=$@
#echo ${args}


for sys in rf ;do
    for i in {1..10};do 
        sed -n '2,$p' ${i}_${sys}_predict.csv | awk -F "," '{print $1,$2}' | sort -n -k2 >${i}_${sys}_tmp
        raws_tmp=($(sed -n '$=' ${i}_${sys}_tmp))
        #for j in {1..10};do
        ############################# line number
        #middellinearray=(575 574 573 572 571 570 569 568 581 580 579 578 577 576 567 566 565 564 563 562)
        middellinearray=${args[@]}
        for j in ${middellinearray[@]};do
            #line_tmp=$(echo ${raws_tmp} ${j} 1 | awk '{print $1-$2+$3}')
            line_tmp=${j}
            sysindex=($(sed -n ''"${line_tmp}"'p' ${i}_${sys}_tmp))
            echo ${sysindex[@]} ${j} ${i}_${sys}
        done
    done
done > top_10_tmp
awk '{print $1}' top_10_tmp | sort -n -k1 | uniq -c | sort -n -k1 -r >top_10_uniqnum_tmp
raws=$(sed -n '$=' top_10_uniqnum_tmp)
for((k=1;k<=${raws};k++));do
    line=($(sed -n ''"${k}"'p' top_10_uniqnum_tmp))
    INDEXCOMP=($(sed -n '2,$p' compernent-index.csv | awk -F "," -v INDEX=${line[1]} '$1==INDEX{print $2,$3,$4,$5,$6}'))
    
    #rf_indexarray=()
    rf_indexarray=($(for i in {1..10};do awk -F "," -v num=${line[1]} '$1==num{print $2}' ${i}_rf_predict.csv; done))
    rf_sum=$(echo ${rf_indexarray[@]} | tr " " + | bc)
    rf_mean=$(echo ${rf_sum} 10.0 | tr " " / | bc)
    
    rbf_indexarray=($(for i in {1..10};do awk -F "," -v num=${line[1]} '$1==num{print $2}' ${i}_rbf_predict.csv; done))    
    rbf_sum=$(echo ${rbf_indexarray[@]} | tr " " + | bc)
    rbf_mean=$(echo ${rbf_sum} 10.0 | tr " " / | bc)

    linear_indexarray=($(for i in {1..10};do awk -F "," -v num=${line[1]} '$1==num{print $2}' ${i}_linear_predict.csv; done))    
    linear_sum=$(echo ${linear_indexarray[@]} | tr " " + | bc)
    linear_mean=$(echo ${linear_sum} 10.0 | tr " " / | bc)


    comp=$(echo ${INDEXCOMP[@]} | sed  's/ /,/g')
    grep "${comp}" first48-1s-2s-unform.v2.csv >/dev/null
    [ $? -ne 0 ] && stat="ok" || stat="exist"
    printf " %s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" ${line[@]} ${INDEXCOMP[@]} ${stat} ${k}  ${rf_mean} rf  ${rbf_mean} rbf ${linear_mean} linear
    #echo ${line[@]} ${INDEXCOMP[@]} ${stat} ${rf_sum} ${rf_mean}
done

