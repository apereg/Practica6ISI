%------------------------------------
% CLASIFICADOR K VECINOS MAS PROXIMOS
%------------------------------------

clear

%% PARTE 1: CARGA CONJUNTO DE DATOS Y PARTICION TRAIN-TEST

load spambase_data.mat;
% X contiene los elementos que se van a estudiar (cada fila corresponde a 
% un vector de caracteristicas)
% Y contiene la clase de cada elemento 

% Numero de elementos del dataset y de variables que tiene cada uno 
[num_patrones, num_variables] = size(X);

% Parametro que indica el porcentaje de elementos que se utilizaran en 
% el conjunto de entrenamiento
p_train = 0.7;

% En la siguiente seccion de codigo se realiza la particion de los datos en 
% entrenamiento y test. Indica lo que realiza cada linea de codigo mediante 
% comentarios.
% ============================================

% Redondea para obtener el numero de elementos que se usaran de
%entrenamiento (70 %)
num_patrones_train = round(p_train*num_patrones); 

% Permuta aleatoriamente el vector num_patrones
ind_permuta = randperm(num_patrones);

% Selecciona los num_patrones_train primeros elementos de la permutacion
% previamente realizada para inds_train y los restantes los almacena en
% inds_test
inds_train = ind_permuta(1:num_patrones_train);
inds_test = ind_permuta(num_patrones_train+1:end);

% Almacena en el vector X_train todas las columnas de las filas de
% inds_train y en Y_train las etiquetas
X_train = X(inds_train, :);
Y_train = Y(inds_train);

% Almacena en el vector X_test todas las columnas de las filas de
% inds_test y en Y_test las etiquetas
X_test= X(inds_test, :);
Y_test = Y(inds_test);

% ============================================

%% PARTE 2: ALGORITMO DE LOS K VECINOS MAS CERCANOS

% La funcion fClassify_kNN ejecuta el algoritmo de kNN. Abrela y completa 
% el codigo
k=3;

Y_test_asig = fClassify_kNN(X_train, Y_train, X_test, k);


%% PARTE 3: EVALUACION DEL RENDIMIENTO DEL CLASIFICADOR

% Muestra matriz de confusion
plotconfusion(Y_test, Y_test_asig);

% Error--> Error global
% ====================== YOUR CODE HERE ======================
TN = sum(Y_test_asig==0 & Y_test==0);
FN = sum(Y_test_asig==1 & Y_test==0);
TP = sum(Y_test_asig==1 & Y_test==1);
FP = sum(Y_test_asig==0 & Y_test==1);
error = (FP+FN)/(TP+TN+FP+FN);
% ============================================================
fprintf('\n******\nError global = %1.4f%% (classification)\n', error*100);

% Tasa de falsa aceptacion
% ====================== YOUR CODE HERE ======================
FPR = FP/(FP+TN);
% ============================================================
fprintf('\n******\nTasa de falsa aceptacion = %1.4f%% (classification)\n', FPR*100);

% Tasa de falso rechazo
% ====================== YOUR CODE HERE ======================
FNR = FN/(TP+FN);
% ============================================================
fprintf('\n******\nTasa de falso rechazo = %1.4f%% (classification)\n', FNR*100);

% Precision
% ====================== YOUR CODE HERE ======================
precision = TP/(TP+FP);
% ============================================================
fprintf('\n******\nPrecision = %1.4f%% (classification)\n', precision*100);

% Recall
% ====================== YOUR CODE HERE ======================

recall = sum(Y_test_asig==1 & Y_test==1)/sum(Y_test==1);

% ============================================================
fprintf('\n******\nRecall = %1.4f%% (classification)\n', recall*100);

