function output = positive_def(array)
    output = array;
    for i = 1:length(array)
        if array(i) < 0
            output(i) = 0;
        end
    end
end