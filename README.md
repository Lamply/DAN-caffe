## DAN-caffe
A Caffe implementations of [Deep Alignment Network](https://github.com/MarekKowalski/DeepAlignmentNetwork). 
- Python custom layer: training and inference
- C++ custom layer: inference in CPU

### Notices
This project is an archived project, which has been stopped maintaining after 2018 and is for reference only.


### Test metrics
Paper:
Stage 1:
Normalization is set to: centers
Failure threshold is set to: 0.08
Processing common subset of the 300W public test set (test sets of LFPW and HELEN)
Average error: 0.0493735770262
Processing challenging subset of the 300W public test set (IBUG dataset)
Average error: 0.0896375119228
Showing results for the entire 300W pulic test set (IBUG dataset, test sets of LFPW and HELEN
Average error: 0.0572627369841
AUC @ 0.08: 0.331357039187
Failure rate: 0.161103047896

Stage 2:
Normalization is set to: centers
Failure threshold is set to: 0.08
Processing common subset of the 300W public test set (test sets of LFPW and HELEN)
Average error: 0.0441422118048
Processing challenging subset of the 300W public test set (IBUG dataset)
Average error: 0.0755380425785
Showing results for the entire 300W pulic test set (IBUG dataset, test sets of LFPW and HELEN
Average error: 0.0502937896778
AUC @ 0.08: 0.390457789066
Failure rate: 0.0812772133527


My:
Stage 1:
Normalization is set to: centers
Failure threshold is set to: 0.08
Processing common subset of the 300W public test set (test sets of LFPW and HELEN)
Average error: 0.0528969781986
Processing challenging subset of the 300W public test set (IBUG dataset)
Average error: 0.0936767290662
Showing results for the entire 300W pulic test set (IBUG dataset, test sets of LFPW and HELEN
Average error: 0.0608872051465
AUC @ 0.08: 0.293641146589
Failure rate: 0.162554426705

Stage 2:
Normalization is set to: centers
Failure threshold is set to: 0.08
Processing common subset of the 300W public test set (test sets of LFPW and HELEN)
Average error: 0.0430267915831
Processing challenging subset of the 300W public test set (IBUG dataset)
Average error: 0.0795358445265
Showing results for the entire 300W pulic test set (IBUG dataset, test sets of LFPW and HELEN
Average error: 0.0501802344675
AUC @ 0.08: 0.398611514272
Failure rate: 0.088534107402

