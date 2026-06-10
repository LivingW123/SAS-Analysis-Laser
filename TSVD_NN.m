% this file calculates the truncated SVD



rng default
fprintf('[1/5] Loading data and building EDRM...\n');

EDRM = x200';
b = x200(:,1);
gp_sz = 2;
size_0 = size(EDRM, 1);
new_EDRM = zeros(size_0, size_0/gp_sz);
for i = 1:size_0/gp_sz
    new_EDRM(:,i) = sum(EDRM(:,gp_sz*(i-1)+1:gp_sz*(i-1)+gp_sz),2);
end
fprintf('[1/5] Done. new_EDRM size: %dx%d\n', size(new_EDRM,1), size(new_EDRM,2));

x_ran = 200/gp_sz;
x = [1:1:x_ran];
x_double = [1:1:20];
%input = 80*normpdf(x,3,2)'+80*normpdf(x,15,2)';
%input = x.^3'.*exp(-0.7*x)'+12*exp(-0.5*x)';
%input1 = (x_double/2).^3'.*exp(-0.7*x_double/2)'+12*exp(-0.5*x_double/2)';


%{
input = 0.000001*exp(-0.5*x)'+0.00001*normpdf(x,15,4)'+0.00002*normpdf(x,80,5)';
theory = new_EDRM * input;
b = theory + (rand(200,1)-0.5).*theory/10;
%}


%error = (b-theory)./theory;
%b = [s1final10028' s2final10028' 0*s3final' 0*s4final' 0*s5final']';
%b = [s1final' s2final' s3final' s4final' s5final1']';
%b = [s1gamma', s2gamma', s3gamma', s4gamma', s5gamma']';
%b = [s1gamma28',s2gamma28',s3gamma28',0*s4final' 0*s5final']';
%b = [s1final1' s2final1' s3final1' s4final1' s5final1']';
%b = [s1gamma26', s2gamma26', s3gamma26', s4gamma26', s5gamma26']';
%b = test_b;
%b = new_EDRM*result1 - new_EDRM*0.00000004*normpdf(x,8,0.1)';
%b = b + (rand(200,1)-0.5).*b/20;


fprintf('[2/5] Computing SVD...\n');
[U,S,V] = svd(new_EDRM);
sizenum = size(S);
sizenum = sizenum(2);
condition_num3 = S(1,1)/S(sizenum,sizenum);
fprintf('[2/5] Done. Condition number: %.4e\n', condition_num3);
uh = pinv(U);
result = zeros(sizenum,1);


num_terms = 7;

c0 = V(:,1:num_terms)' * result;  % project TSVD result onto V basis
%c0 = zeros(num_terms,1);
%c0 = ones(7,1);

constraint = S(6,6);
fprintf('[3/5] Running truncated SVD (%d terms)...\n', num_terms);
for i = 1:num_terms
    %{
    if S(i,i) < constraint
        temp = (uh(i,:)*b/constraint);
    else
        temp = (uh(i,:)*b/S(i,i));
    end
    %}
    if i > 6
        temp = (uh(i,:)*b/constraint);
    else
        temp = (uh(i,:)*b/S(i,i));
    end
    
    %c0(i) = temp;
    %disp(size(temp));
    result = result + V(:,i)*temp(1,1);
end
fprintf('[3/5] Done.\n');



%{
alpha = 10000;
lambda = 1000000;
minfunc = @(x) norm(new_EDRM*x(1:x_ran)-b)^2+alpha^2*sum(exp(abs(diff(x(1:x_ran)))))^2+lambda^2*sum(abs(x(1:x_ran)-result))^2;
%minfunc = @(x) alpha^2*sum(exp(abs(diff(x(1:x_ran)))))^2+lambda^2*sum(abs(x(1:x_ran)-result))^2;
%minfunc = @(x) norm(new_EDRM*x(1:x_ran)-b)^2+x(x_ran+1)^2*sum(exp(abs(diff(x(1:x_ran)))).^6)^2+x(x_ran+2)^2*sum(abs(x(1:x_ran)-result))^2;
%minfunc = @(x) x(1)*exp(-norm(x)^2);
%minfunc = @(x) norm(new_EDRM*x(1:x_ran)-b)^2+lambda^2*norm(x(1:x_ran)-result)^2;
%minimum = minfunc(result);
%disp(abs(diff(result)));
%disp(alpha*sum(abs(diff(x))));


lb = zeros(x_ran+2,1);
ub = ones(x_ran+2,1).*100;
nvars = x_ran;
x_0 = zeros(x_ran+2,1);
%x_1 = x_0(1:x_ran-2);
disp(x_0(1:x_ran));
x_0(1:x_ran) = result;
x_0(x_ran+1:x_ran+2) = [1000000,100000000];
options = optimset("MaxFunEvals",1e12);
options = optimset(options,"MaxIter",1e12);
x = fminsearch(minfunc,x_0,options);
%}

%result2 = cgs(EDRM,b,1e-03,10000000);
%tsvd_nn = @(c,xdata)new_EDRM*positive_def(c(1)*xdata(:,1)+c(2)*xdata(:,2)+c(3)*xdata(:,3)+c(4)*xdata(:,4)+c(5)*xdata(:,5)+c(6)*xdata(:,6));
%tsvd_nn = @(c,xdata)new_EDRM*positive_def(c(1)*xdata(:,1)+c(2)*xdata(:,2)+c(3)*xdata(:,3)+c(4)*xdata(:,4)+c(5)*xdata(:,5));
%tsvd_nn = @(c,xdata)new_EDRM*positive_def(c(1)*xdata(:,1)+c(2)*xdata(:,2)+c(3)*xdata(:,3)+c(4)*xdata(:,4));
tsvd_nn =@(c,xdata)new_EDRM*positive_def(c(1)*xdata(:,1)+c(2)*xdata(:,2)+c(3)*xdata(:,3)+c(4)*xdata(:,4)+c(5)*xdata(:,5)+c(6)*xdata(:,6)+c(7)*xdata(:,7));
%tsvd_nn =@(c,xdata)new_EDRM*positive_def(c(1)*xdata(:,1)+c(2)*xdata(:,2)+c(3)*xdata(:,3)+c(4)*xdata(:,4)+c(5)*xdata(:,5)+c(6)*xdata(:,6)+c(7)*xdata(:,7)+c(8)*xdata(:,8)+c(9)*xdata(:,9));


xdata = 1:1:x_ran;
fprintf('[4/5] Running lsqcurvefit (this may take a while)...\n');
options = optimoptions(@lsqcurvefit,'StepTolerance',1.e-6, 'MaxFunctionEvaluations',1e+5, 'MaxIterations', 1e+4, 'Display','iter');
[c, resnorm, residual, exitflag, output] = lsqcurvefit(tsvd_nn,c0,V,b,[],[],options);
fprintf('[4/5] Done.\n');
fprintf('  Exit flag     : %d (1=converged, 0=max iter hit, <0=failed)\n', exitflag);
fprintf('  Iterations    : %d\n', output.iterations);
fprintf('  Residual norm : %.6e\n', resnorm);
fprintf('  Relative res  : %.6e\n', sqrt(resnorm)/norm(b));
fprintf('  Coefficients  : '); disp(c');



result1 = zeros(sizenum,1);
for i = 1:num_terms
    result1 = result1 + V(:,i)*c(i);
end


result1 = positive_def(result1);
fprintf('[5/5] Done. Plotting result.\n');

%result1 = log(result1);








figure(2);
subplot(2,1,1);
plot(new_EDRM * result1, 'b'); hold on; plot(b, 'r--');
legend('Fitted (EDRM*result1)', 'Measured (b)');
title('Fit vs Data'); xlabel('Detector'); ylabel('Response');
subplot(2,1,2);
plot(residual);
yline(0, 'k--');
title(sprintf('Residuals (norm=%.3e, rel=%.3e)', resnorm, sqrt(resnorm)/norm(b)));
xlabel('Detector'); ylabel('Residual');
saveas(figure(2), 'fit_vs_data.png');
fprintf('Saved fit_vs_data.png\n');

figure(1);hold on;

%a1 = plot(input,'r');M1 = "Input function";
%plot(B);
%a2 = plot(x(1:x_ran),'b');M2 = "Regularized SVD";
%a3 = plot(result,'g');M3 = "TSVD";
%legend([a1,a3],[M1,M3]);
%plot(result);
plot(result1);
%set(gca,'yscale','log');
%plot(V(:,6));
%plot(result2);
%plot(predicted);
%plot(input);
%plot(x(1:x_ran));
hold off;
saveas(figure(1), 'result1.png');
fprintf('Saved result1.png\n');
%disp(norm(new_EDRM*x(1:x_ran)



%disp((result1(3)-result1(2))/4*4);





