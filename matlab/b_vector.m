y = zeros(1, 200);

%{
for i=1:50
    y(i) = x0(i);
   
    %{
    if x(i) <= 0
        y(i) = 0;
    end
    %}
end
%}




%test_b = new_EDRM*x0;
%result2 = result1;
%result2(29:47) = result1(50:68);
%result2(25:43) = result1(29:47);
%result_o0 = result;
%result(7:21) = 0;
test_b2 = b;
test_b1 = new_EDRM*result1;
test_b3 = new_EDRM*predicted;
%test_b3 = new_EDRM*predicted;
figure(2);hold on;
%plot(test_b/7.5,'r');
%plot(test_b2,'r');
a1 = plot(test_b2,'k');M1="Scintillation Light Profile";
a2 = plot(test_b1,'r');M2="TSVD\_NN";
a3 = plot(test_b3,'b');M3="PFF";
%plot(test_b3,'g');
%a4 = plot(b, 'k');M4="input b";
legend([a1,a2,a3],[M1,M2,M3]);
hold off;

disp(norm(test_b2 - b));
disp(norm(test_b1 - b));
%disp(norm(test_b3 - b));
%{
disp(rsquare(test_b,b));
disp(rsquare(b0, b));
%}


figure(3); hold on;
%plot(result2);
plot(result);
%plot(predicted);
set(gca,'yscale','log');
hold off;

exp_result1 = log(result1);
exp_result = log(result);

disp((exp_result1(6)-exp_result1(1))/5);
disp((exp_result(6)-exp_result(1))/5);
%result_original = result1;



