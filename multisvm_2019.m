function result = multisvm_2019(Train_d,Test_d,Labels)

% Train_d = rand(100,100);
% Test_d = Train_d(5,:);
% Labels = 1:100;

% Cls = fitcsvm(Train_d,Labels);
% Class = Cls.predict(Test_d);

% [result] = multisvm(Train_d,Labels,Test_d)


GroupTrain_target = Labels;
TestSet_fea = Test_d;
TrainingSet_fea = Train_d;

u = unique(GroupTrain_target);

num_of_Class = length(u);

zero_mat = zeros(length(TestSet_fea(:,1)),1);

result = zero_mat;

for k = 1 : num_of_Class
    k
    
    G1vAll=(GroupTrain_target == u(k));
     models = fitcsvm(TrainingSet_fea , G1vAll);
     models1{k} = models;
%     models(k) = svmtrain( TrainingSet_fea , G1vAll );

end

for j = 1 : size(TestSet_fea,1)
j
    for k = 1 : num_of_Class
        
        if(models1{k}.predict( TestSet_fea( j,: ) )) 
            
            break;
            
        end
        
    end
    
    result(j) = k;

end

