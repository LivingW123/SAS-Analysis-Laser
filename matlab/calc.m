%comp1=sum(predicted(1:23));
%comp2=sum(predicted(24:60));

comp1=0;
comp2=0;

for idx = 1:23
    comp1 = comp1+idx*predicted(idx);
end

for idx = 24:60
    comp2 = comp2+idx*predicted(idx);
end


ratio = comp2/comp1;

fprintf('comp1 (weighted sum idx 1-23)  : %.6e\n', comp1);
fprintf('comp2 (weighted sum idx 24-60) : %.6e\n', comp2);
fprintf('ratio (comp2/comp1)            : %.6f\n', ratio);



