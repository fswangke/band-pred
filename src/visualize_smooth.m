clear all;
close all;
clc;

load ../data/exp4_range50_rates8_pkts16/data_numpy.mat;
n = 85;
x_raw = trainX_raw(n, :);
x_smooth = trainX_smooth(n, :);
x_send_raw = x_raw(1:128);
x_recv_raw = x_raw(129:end);
x_send_smooth = x_smooth(1:128);
x_recv_smooth = x_smooth(129:end);
idx = 1 : 128;
fig_h = figure;
semilogy(idx, x_send_raw, idx, x_send_smooth);
xlabel('Packet ID'); ylabel('Packet gap (ns)');
legend('Raw', 'Smooth');
print(fig_h, '-dpdf', 'smooth_send.pdf');
print(fig_h, '-dpng', 'smooth_send.png');
close(fig_h);

fig_h = figure;
semilogy(idx, x_recv_raw, idx, x_recv_smooth);
xlabel('Packet ID'); ylabel('Packet gap (ns)');
legend('Raw', 'Smooth');
print(fig_h, '-dpdf', 'smooth_recv.pdf');
print(fig_h, '-dpng', 'smooth_recv.png');
close(fig_h);

% FFT
x_send_raw_fft = fft(x_send_raw);
x_send_smooth_fft = fft(x_send_smooth);
fig_h = figure;
scatter(real(x_send_raw_fft), imag(x_send_raw_fft), [], 'r');
hold on;
scatter(real(x_send_smooth_fft), imag(x_send_smooth_fft), [], 'b');
xlabel('Real part');
ylabel('Imaginary part');
legend('Raw', 'Smooth');
print(fig_h, '-dpdf', 'fft_send.pdf');
print(fig_h, '-dpng', 'fft_send.png');
close(fig_h);

x_recv_raw_fft = fft(x_recv_raw);
x_recv_smooth_fft = fft(x_recv_smooth);
fig_h = figure;
scatter(real(x_recv_raw_fft), imag(x_recv_raw_fft), [], 'r');
hold on;
scatter(real(x_recv_smooth_fft), imag(x_recv_smooth_fft), [], 'b');
xlabel('Real part');
ylabel('Imaginary part');
legend('Raw', 'Smooth');
print(fig_h, '-dpdf', 'fft_recv.pdf');
print(fig_h, '-dpng', 'fft_recv.png');
close(fig_h);
