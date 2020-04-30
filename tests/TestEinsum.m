classdef TestEinsum < matlab.unittest.TestCase
    methods(Test)
        function basic_operations(testCase)
            a = permute(reshape(0:5, [3,2]), [2, 1]);  % Matrix in C-Order

            % Matrix Transpose
            testCase.verifyEqual(einsum('ij->ji', a), a.');

            % Sum
            testCase.verifyEqual(einsum('ij->', a), sum(a(:)));
            
            % Column Sum
            testCase.verifyEqual(einsum('ij->j', a), sum(a, 1)');
            
            % Row Sum
            testCase.verifyEqual(einsum('ij->i', a), sum(a, 2));
            
            % Matrix-Vector Multiplication and Sum
            b = (0:2);
            testCase.verifyEqual(einsum('ik,jk->i', a, b), sum(a.*b, 2));            

            % Matrix-Matrix Multiplication
            b = permute(reshape(0:14, [5,3]), [2, 1]);  % Matrix in C-Order
            testCase.verifyEqual(einsum('ik,kj->ij', a, b), a*b);
            
            % Vector Dot Product
            a = 0:2;
            b = 3:5;
            testCase.verifyEqual(einsum('ij,ij->', a, b), dot(a,b));

            % Matrix Dot Product
            a = permute(reshape(0:5, [3,2]), [2, 1]);  % Matrix in C-Order
            b = permute(reshape(6:11, [3,2]), [2, 1]);  % Matrix in C-Order
            testCase.verifyEqual(einsum('ij,ij->', a, b), sum(dot(a,b)));

            % Hadamard Product
            testCase.verifyEqual(einsum('ij,ij->ij', a, b), a.*b);
            
            % Outer Product
            a = 0:2;
            b = 3:6;
            testCase.verifyEqual(einsum('ij,ji->ij', a', b'), a'*b);            
        end
        function check_einsum_sums(testCase)
            % sum(a, axis=-1)
            for n = 2:17
                a = reshape(1:(2*3*n), [2, 3, n]);
                testCase.verifyEqual(einsum('...i->...', a), sum(a, 3));
            end
            
            % sum(a, axis=0)
            for n = 2:17
                a = reshape(1:(2*n), [2, n]);
                testCase.verifyEqual(einsum('i...->...', a), sum(a, 1)');
            end
            for n = 2:17
                a = reshape(1:(2*3*n), [2, 3, n]);
                testCase.verifyEqual(einsum('i...->...', a), squeeze(sum(a, 1)));
            end
            
            % tensordot(a, b)
            a = reshape(0:59, [3, 4, 5]);
            b = reshape(0:23, [4, 3, 2]);
            c = [[  440.,  1232.]; [ 1232.,  3752.]; [ 2024.,  6272.]; [ 2816.,  8792.]; [ 3608., 11312.]];
            testCase.verifyEqual(einsum('ijk, jil -> kl', a, b), c);
            
            % singleton dimension broadcast
            p = ones(10,2);
            q = ones(1,2);
            testCase.verifyEqual(einsum('ij,ij->j', p, q), 10*ones(2,1));            
        end
        function test_einsum_broadcast(testCase)
            a = ones(1, 2);
            b = ones(2, 2, 1);
            testCase.verifyEqual(einsum('ij...,j...->i...', a, b), [2, 2])

            A = reshape(0:11, [3,4])';
            B = reshape(0:5, [2,3])';
            ref = einsum('ik,kj->ij', A, B);
            testCase.verifyEqual(einsum('ik...,k...->i...', A, B), ref)
            testCase.verifyEqual(einsum('ik...,...kj->i...j', A, B), ref)
            testCase.verifyEqual(einsum('ik,k...->i...', A, B), ref)            
        end
    end  % methods
end  % classef
