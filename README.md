# nccl-tests

Modify run_theta.sh as desired and then run the following commands on theta:
```
ssh thetagpusn1
qsub -I -n 2 -t 10 -q full-node -A dist_relational_alg --attrs filesystems=home,grand,theta-fs0
cd <nccl-tests-path>/
./run_theta.sh
```
