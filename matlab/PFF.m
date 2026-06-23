% This script implements parametrized forward folding method using
% chebyshev polynomials

rng default
EDRM = x200';
gp_sz = 2;
size_0 = size(EDRM, 1);
new_EDRM = zeros(size_0, size_0/gp_sz);
for i = 1:size_0/gp_sz
    new_EDRM(:,i) = sum(EDRM(:,gp_sz*(i-1)+1:gp_sz*(i-1)+gp_sz),2);
end
x_ran = 200/gp_sz;
x = [1:1:x_ran];
%b = [s1gamma26', s2gamma26', s3gamma26', s4gamma26', s5gamma26']';

%b = [s1final' s2final' s3final' s4final' 0 0 0 0 0 0 0 0]';
%{
input = 0.0001*exp(-0.1*x)'+0.0005*normpdf(x,9,1)'+0.001*normpdf(x,40,10)';
theory = new_EDRM * input;
b = theory + (rand(200,1)-0.5).*theory/10;
xdata = 1:1:x_ran;
%}
%f_E_process = @(x,xdata)new_EDRM*(x(1)*chebyshevT(0,xdata)'+x(2)*chebyshevT(1,xdata)'+x(3)*chebyshevT(2,xdata)'+x(4)*chebyshevT(3,xdata)'...
%    +x(5)*chebyshevT(4,xdata)');

%f_E_process = @(x,xdata)new_EDRM*((x(1)*xdata./xdata)'+x(2)*xdata'+x(3)*xdata.^2'+x(4)*xdata.^3'+x(5)*xdata.^4')-b;

%f_E_process = @(x,xdata)new_EDRM*((x(1)*xdata./xdata)'+x(2)*xdata'+x(3)*xdata.^2'+x(4)*xdata.^3');


%f_E_process = @(x,xdata)new_EDRM*(x(1)*chebyshevT(0,xdata)'+x(2)*chebyshevT(1,xdata)'+x(3)*chebyshevT(2,xdata)'+x(4)*chebyshevT(3,xdata)');
%f_E_process = @(x,xdata)new_EDRM*(x(1)*exp(-x(2)*xdata)'+x(3)*normpdf(xdata,x(4),x(5))'+x(6)*normpdf(xdata,x(7),x(8))');
f_E_process = @(x,xdata)new_EDRM*(x(1)*exp(-x(2)*xdata)'+x(3)*exp(-(xdata-x(4)).^2./(x(5)*xdata+x(6)./xdata))');%index 1
%f_E_process = @(x,xdata)new_EDRM*(x(1)*exp(-x(2)*xdata)'+x(3)*xdata.^x(4)*exp(-x(5)*xdata)');

%lb = [0 0 0 0 1 0];%for index 1
%ub = [Inf Inf Inf Inf Inf Inf];%for index1
% import the experiment data


lb = [-Inf -Inf -Inf -Inf -Inf -Inf];
ub = [Inf Inf Inf Inf Inf Inf];
%x0 = [0.00000001 0.43 0 34 1.8 10];%this is for index1
%x0 = [0.0000015 0.3 0.0000005 30 0.1 3];
x0 = [0.000001 1 0.000001 33 0.015 15];


%Gaussian fit parameters
%{
lb = [0 0 0 0 5];
ub = [0.000001 0.6 0.0001 42 15];
% import the experiment data
x0 = [0.00000025 0.2 0.0000008 35 8];
%}



options = optimoptions(@lsqcurvefit,'StepTolerance',1e-12);

x = lsqcurvefit(f_E_process,x0,xdata,b,lb,ub);

%f_E = @(x,xdata)x(1)*chebyshevT(0,xdata)'+x(2)*chebyshevT(1,xdata)'+x(3)*chebyshevT(2,xdata)'+x(4)*chebyshevT(3,xdata)'...
%    +x(5)*chebyshevT(4,xdata)';
%f_E = @(x,xdata)(x(1)*xdata./xdata)'+x(2)*xdata'+x(3)*xdata.^2'+x(4)*xdata.^3'+x(5)*xdata.^4';
%f_E = @(x,xdata)(x(1)*xdata./xdata)'+x(2)*xdata'+x(3)*xdata.^2'+x(4)*xdata.^3';
%f_E = @(x,xdata)x(1)*chebyshevT(0,xdata)'+x(2)*chebyshevT(1,xdata)'+x(3)*chebyshevT(2,xdata)'+x(4)*chebyshevT(3,xdata)';
%f_E = @(x,xdata)x(1)*exp(x(2)*xdata)'+x(3)*exp(x(4)*xdata)';
%f_E = @(x,xdata)x(1)*exp(-x(2)*xdata)'+x(3)*normpdf(xdata,x(4),x(5))'+x(6)*normpdf(xdata,x(7),x(8))';
%f_E = @(x,xdata)x(1)*exp(-x(2)*xdata)'+x(3)*normpdf(xdata,x(4),x(5))';
f_E = @(x,xdata)x(1)*exp(-x(2)*xdata)'+x(3)*exp(-(xdata-x(4)).^2./(x(5)*xdata+x(6)./xdata))';


figure(1);hold on;
%plot(input);
plot(f_E(x,xdata),'LineWidth',2);
%plot(result1);
%plot(zeros(1,20));
dim = [.6 .6 .6 .6];
str={'$$f(x) = a_1 e^{-a_2 x}+a_3 e^{-\frac{(x-a_4)^2}{a_5 x + \frac{a_6}{x}}}$$'};
annotation('textbox','Position',[.4 .4 .4 .4],'FontSize',20,'interpreter','latex','String',str,'FitBoxToText','on');
hold off;


predicted = f_E(x,xdata);

figure(2);hold on;
a1 = plot(predicted); a2 = plot(result1);
M1 = "PFF"; M2 = "TSVD\_NN";
legend([a1,a2],[M1,M2]);



%{
for i = 1:length(predicted)
    if predicted(i) < 0
        predicted(i) = 0;
    end
end
figure(2);hold on;
set(gca, 'YScale', 'log');
plot(predicted);
%plot(input);
hold off;

%}

disp(f_E_process(x,xdata)-b);