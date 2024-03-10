% Caffe curve print and analysis

%% DAN stage 2
system('./get_loss.sh -f log/vgglike_s2.log');
x = caffe_log('tmp/train_loss.log');
y = caffe_log('tmp/test_loss.log');
xa = caffe_average(x, 100, 60960, 24);
plot_train_val(x, y);

system('./get_loss.sh -f log/stage2_trainlog_success');
x = caffe_log('tmp/train_loss.log');
y = caffe_log('tmp/test_loss.log');
xa = caffe_average(x, 40, 60960, 12);
plot_train_val(x, y, 1);
