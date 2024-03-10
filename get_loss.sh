#!/bin/sh

while getopts 'f:c' opt; 
do	
    case ${opt} in
        f)
            filename="${OPTARG}";;
	    c) 
	        contiFlag="true";;
        ?)
            echo "Usage: `basename $0` -f filename [-c]"
	    exit 1
    esac
done
echo ${contiFlag}

if [ -n "$contiFlag" ]; then
    cat ${filename} | grep "Test net output #0: " | awk '{printf("%s,",$11)}' >> tmp/test_loss.log
    cat ${filename} | grep ", loss = " | awk '{printf("%s,",$13)}' >> tmp/train_loss.log
else
    cat ${filename} | grep "Test net output #0: " | awk '{printf("%s,",$11)}' > tmp/test_loss.log
    cat ${filename} | grep ", loss = " | awk '{printf("%s,",$13)}' > tmp/train_loss.log
fi


 
