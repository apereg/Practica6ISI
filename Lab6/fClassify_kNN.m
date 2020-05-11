function Y_assign = fClassify_kNN(X_train, Y_train, X_test, k)
% This function implements the kNN classification algorithm with the
% eucludean distance
%
% INPUT
%   - X_train: Matrix (n_train x n), where n_Train is the number of 
%   training elements and n is the number of features (the length of the 
%   feature vector)
%   - Y_train: The classes of the elements in the training set. It is a
%   vector of length n_train with the number of the class.
%   - X_test: matrix (n_test x n), where n_test is the number of elements 
%   in the test set and n is the number of features (the length of the 
%   feature vector).
%   - k: Number of nearest neighbours to consider in order to make an
%   assignation
%
% OUTPUT
%   A vector with length n_test, with the classess assigned by the algorithm 
%   to the elements in the training set.
%

    numElemTest = size(X_test, 1);
    numElemTrain = size(X_train, 1);

    % Allocate space for the output
    Y_assign = zeros(1, numElemTest);
    umbral = fix(k/2);
    
    % for each element in the test set...
    for i=1:numElemTest
        
        x_test_i = X_test(i,:);
        
        % 1 - Compute the Euclidean distance of the i-th test element to 
        % all the training elements
        % ====================== YOUR CODE HERE ======================
        distances = zeros(2, numElemTrain);
        % Se almacena en la matriz las distancias a cada punto y su indice
        % para luego localizarlos
        for j=1:numElemTrain
            distances(1,j) = pdist2(x_test_i,X_train(j, :),'euclidean');
            distances(2,j) = j;
        end
        % ============================================================
        
        % 2 - Order distances in ascending order and use the indices of the 
        % ordering
        % ====================== YOUR CODE HERE ======================
        k_values = sort(distances,2, 'ascend');
        % ============================================================

        % 3 - Take the k first classes of the training set
        % ====================== YOUR CODE HERE ======================
        k_values = k_values(1:k);
        for j=1:k        
            for m=1:numElemTrain
                % Se busca el valor en la matriz distance y se coge el
                % indice para poder localizar su resultado real en Y_Train
                if(distances(1,m) == k_values(j))
                    k_values(j) = distances(2,m);
                    break;
                end
            end
        end
        % ============================================================
        
        % 4 - Assign to the i-th element the most frequent class 
        % ====================== YOUR CODE HERE ======================
        numberZeros = 0;
        % Se calcula el numero de valores 0 entre los k vecinos
        for z=1:k
            if(Y_train(k_values(z)) == 0)
                numberZeros = numberZeros+1;
            end
        end
        
        % Si los valores 0 son mas de la mitad se asigna al valor cero, si
        % no al 1
        if(numberZeros > umbral)
            Y_assign(i) = 0;
        else
            Y_assign(i) = 1;
        end
        % ============================================================
    end

end

