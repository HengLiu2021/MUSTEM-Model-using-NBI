clc
clear


syms Theta [6 1] real


for i = 1:7
    for j = i:7
        %% obtain loss function, f_prime, and f_prime_prime
        switch i
            case 1
                switch j-i
                    case 0
                        Pij = exp(-Theta(i));
                    case 1
                        A = Theta(i)/(Theta(i+1)-Theta(i))*exp(-Theta(i));
                        B = Theta(i)/(Theta(i)-Theta(i+1))*exp(-Theta(i+1));
                        Pij = A+B;
                    case 2
                        A = Theta(i)/(Theta(i+1)-Theta(i))*Theta(i+1)/(Theta(i+2)-Theta(i))*exp(-Theta(i));
                        B = Theta(i)/(Theta(i)-Theta(i+1))*Theta(i+1)/(Theta(i+2)-Theta(i+1))*exp(-Theta(i+1));
                        C = Theta(i)/(Theta(i)-Theta(i+2))*Theta(i+1)/(Theta(i+1)-Theta(i+2))*exp(-Theta(i+2));
                        Pij = A+B+C;
                    case 3
                        A = Theta(i)/(Theta(i+1)-Theta(i))*Theta(i+1)/(Theta(i+2)-Theta(i))*Theta(i+2)/(Theta(i+3)-Theta(i))*exp(-Theta(i));
                        B = Theta(i)/(Theta(i)-Theta(i+1))*Theta(i+1)/(Theta(i+2)-Theta(i+1))*Theta(i+2)/(Theta(i+3)-Theta(i+1))*exp(-Theta(i+1));
                        C = Theta(i)/(Theta(i)-Theta(i+2))*Theta(i+1)/(Theta(i+1)-Theta(i+2))*Theta(i+2)/(Theta(i+3)-Theta(i+2))*exp(-Theta(i+2));
                        D = Theta(i)/(Theta(i)-Theta(i+3))*Theta(i+1)/(Theta(i+1)-Theta(i+3))*Theta(i+2)/(Theta(i+2)-Theta(i+3))*exp(-Theta(i+3));
                        Pij = A+B+C+D;
                    case 4
                        A = Theta(i)/(Theta(i+1)-Theta(i))*Theta(i+1)/(Theta(i+2)-Theta(i))*Theta(i+2)/(Theta(i+3)-Theta(i))*Theta(i+3)/(Theta(i+4)-Theta(i))*exp(-Theta(i));
                        B = Theta(i)/(Theta(i)-Theta(i+1))*Theta(i+1)/(Theta(i+2)-Theta(i+1))*Theta(i+2)/(Theta(i+3)-Theta(i+1))*Theta(i+3)/(Theta(i+4)-Theta(i+1))*exp(-Theta(i+1));
                        C = Theta(i)/(Theta(i)-Theta(i+2))*Theta(i+1)/(Theta(i+1)-Theta(i+2))*Theta(i+2)/(Theta(i+3)-Theta(i+2))*Theta(i+3)/(Theta(i+4)-Theta(i+2))*exp(-Theta(i+2));
                        D = Theta(i)/(Theta(i)-Theta(i+3))*Theta(i+1)/(Theta(i+1)-Theta(i+3))*Theta(i+2)/(Theta(i+2)-Theta(i+3))*Theta(i+3)/(Theta(i+4)-Theta(i+3))*exp(-Theta(i+3));
                        E = Theta(i)/(Theta(i)-Theta(i+4))*Theta(i+1)/(Theta(i+1)-Theta(i+4))*Theta(i+2)/(Theta(i+2)-Theta(i+4))*Theta(i+3)/(Theta(i+3)-Theta(i+4))*exp(-Theta(i+4));
                        Pij = A+B+C+D+E;
                    case 5
                        A = Theta(i)/(Theta(i+1)-Theta(i))*Theta(i+1)/(Theta(i+2)-Theta(i))*Theta(i+2)/(Theta(i+3)-Theta(i))*Theta(i+3)/(Theta(i+4)-Theta(i))*Theta(i+4)/(Theta(i+5)-Theta(i))*exp(-Theta(i));
                        B = Theta(i)/(Theta(i)-Theta(i+1))*Theta(i+1)/(Theta(i+2)-Theta(i+1))*Theta(i+2)/(Theta(i+3)-Theta(i+1))*Theta(i+3)/(Theta(i+4)-Theta(i+1))*Theta(i+4)/(Theta(i+5)-Theta(i+1))*exp(-Theta(i+1));
                        C = Theta(i)/(Theta(i)-Theta(i+2))*Theta(i+1)/(Theta(i+1)-Theta(i+2))*Theta(i+2)/(Theta(i+3)-Theta(i+2))*Theta(i+3)/(Theta(i+4)-Theta(i+2))*Theta(i+4)/(Theta(i+5)-Theta(i+2))*exp(-Theta(i+2));
                        D = Theta(i)/(Theta(i)-Theta(i+3))*Theta(i+1)/(Theta(i+1)-Theta(i+3))*Theta(i+2)/(Theta(i+2)-Theta(i+3))*Theta(i+3)/(Theta(i+4)-Theta(i+3))*Theta(i+4)/(Theta(i+5)-Theta(i+3))*exp(-Theta(i+3));
                        E = Theta(i)/(Theta(i)-Theta(i+4))*Theta(i+1)/(Theta(i+1)-Theta(i+4))*Theta(i+2)/(Theta(i+2)-Theta(i+4))*Theta(i+3)/(Theta(i+3)-Theta(i+4))*Theta(i+4)/(Theta(i+5)-Theta(i+4))*exp(-Theta(i+4));
                        F = Theta(i)/(Theta(i)-Theta(i+5))*Theta(i+1)/(Theta(i+1)-Theta(i+5))*Theta(i+2)/(Theta(i+2)-Theta(i+5))*Theta(i+3)/(Theta(i+3)-Theta(i+5))*Theta(i+4)/(Theta(i+4)-Theta(i+5))*exp(-Theta(i+5));
                        Pij = A+B+C+D+E+F;
                    case 6
                        A0 = exp(-Theta(i));
                        A1 = Theta(i)/(Theta(i+1)-Theta(i))*exp(-Theta(i));
                        B1 = Theta(i)/(Theta(i)-Theta(i+1))*exp(-Theta(i+1));
                        A2 = Theta(i)/(Theta(i+1)-Theta(i))*Theta(i+1)/(Theta(i+2)-Theta(i))*exp(-Theta(i));
                        B2 = Theta(i)/(Theta(i)-Theta(i+1))*Theta(i+1)/(Theta(i+2)-Theta(i+1))*exp(-Theta(i+1));
                        C2 = Theta(i)/(Theta(i)-Theta(i+2))*Theta(i+1)/(Theta(i+1)-Theta(i+2))*exp(-Theta(i+2));
                        A3 = Theta(i)/(Theta(i+1)-Theta(i))*Theta(i+1)/(Theta(i+2)-Theta(i))*Theta(i+2)/(Theta(i+3)-Theta(i))*exp(-Theta(i));
                        B3 = Theta(i)/(Theta(i)-Theta(i+1))*Theta(i+1)/(Theta(i+2)-Theta(i+1))*Theta(i+2)/(Theta(i+3)-Theta(i+1))*exp(-Theta(i+1));
                        C3 = Theta(i)/(Theta(i)-Theta(i+2))*Theta(i+1)/(Theta(i+1)-Theta(i+2))*Theta(i+2)/(Theta(i+3)-Theta(i+2))*exp(-Theta(i+2));
                        D3 = Theta(i)/(Theta(i)-Theta(i+3))*Theta(i+1)/(Theta(i+1)-Theta(i+3))*Theta(i+2)/(Theta(i+2)-Theta(i+3))*exp(-Theta(i+3));
                        A4 = Theta(i)/(Theta(i+1)-Theta(i))*Theta(i+1)/(Theta(i+2)-Theta(i))*Theta(i+2)/(Theta(i+3)-Theta(i))*Theta(i+3)/(Theta(i+4)-Theta(i))*exp(-Theta(i));
                        B4 = Theta(i)/(Theta(i)-Theta(i+1))*Theta(i+1)/(Theta(i+2)-Theta(i+1))*Theta(i+2)/(Theta(i+3)-Theta(i+1))*Theta(i+3)/(Theta(i+4)-Theta(i+1))*exp(-Theta(i+1));
                        C4 = Theta(i)/(Theta(i)-Theta(i+2))*Theta(i+1)/(Theta(i+1)-Theta(i+2))*Theta(i+2)/(Theta(i+3)-Theta(i+2))*Theta(i+3)/(Theta(i+4)-Theta(i+2))*exp(-Theta(i+2));
                        D4 = Theta(i)/(Theta(i)-Theta(i+3))*Theta(i+1)/(Theta(i+1)-Theta(i+3))*Theta(i+2)/(Theta(i+2)-Theta(i+3))*Theta(i+3)/(Theta(i+4)-Theta(i+3))*exp(-Theta(i+3));
                        E4 = Theta(i)/(Theta(i)-Theta(i+4))*Theta(i+1)/(Theta(i+1)-Theta(i+4))*Theta(i+2)/(Theta(i+2)-Theta(i+4))*Theta(i+3)/(Theta(i+3)-Theta(i+4))*exp(-Theta(i+4));
                        A5 = Theta(i)/(Theta(i+1)-Theta(i))*Theta(i+1)/(Theta(i+2)-Theta(i))*Theta(i+2)/(Theta(i+3)-Theta(i))*Theta(i+3)/(Theta(i+4)-Theta(i))*Theta(i+4)/(Theta(i+5)-Theta(i))*exp(-Theta(i));
                        B5 = Theta(i)/(Theta(i)-Theta(i+1))*Theta(i+1)/(Theta(i+2)-Theta(i+1))*Theta(i+2)/(Theta(i+3)-Theta(i+1))*Theta(i+3)/(Theta(i+4)-Theta(i+1))*Theta(i+4)/(Theta(i+5)-Theta(i+1))*exp(-Theta(i+1));
                        C5 = Theta(i)/(Theta(i)-Theta(i+2))*Theta(i+1)/(Theta(i+1)-Theta(i+2))*Theta(i+2)/(Theta(i+3)-Theta(i+2))*Theta(i+3)/(Theta(i+4)-Theta(i+2))*Theta(i+4)/(Theta(i+5)-Theta(i+2))*exp(-Theta(i+2));
                        D5 = Theta(i)/(Theta(i)-Theta(i+3))*Theta(i+1)/(Theta(i+1)-Theta(i+3))*Theta(i+2)/(Theta(i+2)-Theta(i+3))*Theta(i+3)/(Theta(i+4)-Theta(i+3))*Theta(i+4)/(Theta(i+5)-Theta(i+3))*exp(-Theta(i+3));
                        E5 = Theta(i)/(Theta(i)-Theta(i+4))*Theta(i+1)/(Theta(i+1)-Theta(i+4))*Theta(i+2)/(Theta(i+2)-Theta(i+4))*Theta(i+3)/(Theta(i+3)-Theta(i+4))*Theta(i+4)/(Theta(i+5)-Theta(i+4))*exp(-Theta(i+4));
                        F5 = Theta(i)/(Theta(i)-Theta(i+5))*Theta(i+1)/(Theta(i+1)-Theta(i+5))*Theta(i+2)/(Theta(i+2)-Theta(i+5))*Theta(i+3)/(Theta(i+3)-Theta(i+5))*Theta(i+4)/(Theta(i+4)-Theta(i+5))*exp(-Theta(i+5));
                        Pij = 1-(A0+A1+B1+A2+B2+C2+A3+B3+C3+D3+A4+B4+C4+D4+E4+A5+B5+C5+D5+E5+F5);
                end

            case 2
                switch j-i
                    case 0
                        Pij = exp(-Theta(i));
                    case 1
                        A = Theta(i)/(Theta(i+1)-Theta(i))*exp(-Theta(i));
                        B = Theta(i)/(Theta(i)-Theta(i+1))*exp(-Theta(i+1));
                        Pij = A+B;
                    case 2
                        A = Theta(i)/(Theta(i+1)-Theta(i))*Theta(i+1)/(Theta(i+2)-Theta(i))*exp(-Theta(i));
                        B = Theta(i)/(Theta(i)-Theta(i+1))*Theta(i+1)/(Theta(i+2)-Theta(i+1))*exp(-Theta(i+1));
                        C = Theta(i)/(Theta(i)-Theta(i+2))*Theta(i+1)/(Theta(i+1)-Theta(i+2))*exp(-Theta(i+2));
                        Pij = A+B+C;
                    case 3
                        A = Theta(i)/(Theta(i+1)-Theta(i))*Theta(i+1)/(Theta(i+2)-Theta(i))*Theta(i+2)/(Theta(i+3)-Theta(i))*exp(-Theta(i));
                        B = Theta(i)/(Theta(i)-Theta(i+1))*Theta(i+1)/(Theta(i+2)-Theta(i+1))*Theta(i+2)/(Theta(i+3)-Theta(i+1))*exp(-Theta(i+1));
                        C = Theta(i)/(Theta(i)-Theta(i+2))*Theta(i+1)/(Theta(i+1)-Theta(i+2))*Theta(i+2)/(Theta(i+3)-Theta(i+2))*exp(-Theta(i+2));
                        D = Theta(i)/(Theta(i)-Theta(i+3))*Theta(i+1)/(Theta(i+1)-Theta(i+3))*Theta(i+2)/(Theta(i+2)-Theta(i+3))*exp(-Theta(i+3));
                        Pij = A+B+C+D;
                    case 4
                        A = Theta(i)/(Theta(i+1)-Theta(i))*Theta(i+1)/(Theta(i+2)-Theta(i))*Theta(i+2)/(Theta(i+3)-Theta(i))*Theta(i+3)/(Theta(i+4)-Theta(i))*exp(-Theta(i));
                        B = Theta(i)/(Theta(i)-Theta(i+1))*Theta(i+1)/(Theta(i+2)-Theta(i+1))*Theta(i+2)/(Theta(i+3)-Theta(i+1))*Theta(i+3)/(Theta(i+4)-Theta(i+1))*exp(-Theta(i+1));
                        C = Theta(i)/(Theta(i)-Theta(i+2))*Theta(i+1)/(Theta(i+1)-Theta(i+2))*Theta(i+2)/(Theta(i+3)-Theta(i+2))*Theta(i+3)/(Theta(i+4)-Theta(i+2))*exp(-Theta(i+2));
                        D = Theta(i)/(Theta(i)-Theta(i+3))*Theta(i+1)/(Theta(i+1)-Theta(i+3))*Theta(i+2)/(Theta(i+2)-Theta(i+3))*Theta(i+3)/(Theta(i+4)-Theta(i+3))*exp(-Theta(i+3));
                        E = Theta(i)/(Theta(i)-Theta(i+4))*Theta(i+1)/(Theta(i+1)-Theta(i+4))*Theta(i+2)/(Theta(i+2)-Theta(i+4))*Theta(i+3)/(Theta(i+3)-Theta(i+4))*exp(-Theta(i+4));
                        Pij = A+B+C+D+E;
                    case 5
                        A0 = exp(-Theta(i));
                        A1 = Theta(i)/(Theta(i+1)-Theta(i))*exp(-Theta(i));
                        B1 = Theta(i)/(Theta(i)-Theta(i+1))*exp(-Theta(i+1));
                        A2 = Theta(i)/(Theta(i+1)-Theta(i))*Theta(i+1)/(Theta(i+2)-Theta(i))*exp(-Theta(i));
                        B2 = Theta(i)/(Theta(i)-Theta(i+1))*Theta(i+1)/(Theta(i+2)-Theta(i+1))*exp(-Theta(i+1));
                        C2 = Theta(i)/(Theta(i)-Theta(i+2))*Theta(i+1)/(Theta(i+1)-Theta(i+2))*exp(-Theta(i+2));
                        A3 = Theta(i)/(Theta(i+1)-Theta(i))*Theta(i+1)/(Theta(i+2)-Theta(i))*Theta(i+2)/(Theta(i+3)-Theta(i))*exp(-Theta(i));
                        B3 = Theta(i)/(Theta(i)-Theta(i+1))*Theta(i+1)/(Theta(i+2)-Theta(i+1))*Theta(i+2)/(Theta(i+3)-Theta(i+1))*exp(-Theta(i+1));
                        C3 = Theta(i)/(Theta(i)-Theta(i+2))*Theta(i+1)/(Theta(i+1)-Theta(i+2))*Theta(i+2)/(Theta(i+3)-Theta(i+2))*exp(-Theta(i+2));
                        D3 = Theta(i)/(Theta(i)-Theta(i+3))*Theta(i+1)/(Theta(i+1)-Theta(i+3))*Theta(i+2)/(Theta(i+2)-Theta(i+3))*exp(-Theta(i+3));
                        A4 = Theta(i)/(Theta(i+1)-Theta(i))*Theta(i+1)/(Theta(i+2)-Theta(i))*Theta(i+2)/(Theta(i+3)-Theta(i))*Theta(i+3)/(Theta(i+4)-Theta(i))*exp(-Theta(i));
                        B4 = Theta(i)/(Theta(i)-Theta(i+1))*Theta(i+1)/(Theta(i+2)-Theta(i+1))*Theta(i+2)/(Theta(i+3)-Theta(i+1))*Theta(i+3)/(Theta(i+4)-Theta(i+1))*exp(-Theta(i+1));
                        C4 = Theta(i)/(Theta(i)-Theta(i+2))*Theta(i+1)/(Theta(i+1)-Theta(i+2))*Theta(i+2)/(Theta(i+3)-Theta(i+2))*Theta(i+3)/(Theta(i+4)-Theta(i+2))*exp(-Theta(i+2));
                        D4 = Theta(i)/(Theta(i)-Theta(i+3))*Theta(i+1)/(Theta(i+1)-Theta(i+3))*Theta(i+2)/(Theta(i+2)-Theta(i+3))*Theta(i+3)/(Theta(i+4)-Theta(i+3))*exp(-Theta(i+3));
                        E4 = Theta(i)/(Theta(i)-Theta(i+4))*Theta(i+1)/(Theta(i+1)-Theta(i+4))*Theta(i+2)/(Theta(i+2)-Theta(i+4))*Theta(i+3)/(Theta(i+3)-Theta(i+4))*exp(-Theta(i+4));
                        Pij = 1-(A0+A1+B1+A2+B2+C2+A3+B3+C3+D3+A4+B4+C4+D4+E4);
                end
            case 3
                switch j-i
                    case 0
                        Pij = exp(-Theta(i));
                    case 1
                        A = Theta(i)/(Theta(i+1)-Theta(i))*exp(-Theta(i));
                        B = Theta(i)/(Theta(i)-Theta(i+1))*exp(-Theta(i+1));
                        Pij = A+B;
                    case 2
                        A = Theta(i)/(Theta(i+1)-Theta(i))*Theta(i+1)/(Theta(i+2)-Theta(i))*exp(-Theta(i));
                        B = Theta(i)/(Theta(i)-Theta(i+1))*Theta(i+1)/(Theta(i+2)-Theta(i+1))*exp(-Theta(i+1));
                        C = Theta(i)/(Theta(i)-Theta(i+2))*Theta(i+1)/(Theta(i+1)-Theta(i+2))*exp(-Theta(i+2));
                        Pij = A+B+C;
                    case 3
                        A = Theta(i)/(Theta(i+1)-Theta(i))*Theta(i+1)/(Theta(i+2)-Theta(i))*Theta(i+2)/(Theta(i+3)-Theta(i))*exp(-Theta(i));
                        B = Theta(i)/(Theta(i)-Theta(i+1))*Theta(i+1)/(Theta(i+2)-Theta(i+1))*Theta(i+2)/(Theta(i+3)-Theta(i+1))*exp(-Theta(i+1));
                        C = Theta(i)/(Theta(i)-Theta(i+2))*Theta(i+1)/(Theta(i+1)-Theta(i+2))*Theta(i+2)/(Theta(i+3)-Theta(i+2))*exp(-Theta(i+2));
                        D = Theta(i)/(Theta(i)-Theta(i+3))*Theta(i+1)/(Theta(i+1)-Theta(i+3))*Theta(i+2)/(Theta(i+2)-Theta(i+3))*exp(-Theta(i+3));
                        Pij = A+B+C+D;
                    case 4
                        A0 = exp(-Theta(i));
                        A1 = Theta(i)/(Theta(i+1)-Theta(i))*exp(-Theta(i));
                        B1 = Theta(i)/(Theta(i)-Theta(i+1))*exp(-Theta(i+1));
                        A2 = Theta(i)/(Theta(i+1)-Theta(i))*Theta(i+1)/(Theta(i+2)-Theta(i))*exp(-Theta(i));
                        B2 = Theta(i)/(Theta(i)-Theta(i+1))*Theta(i+1)/(Theta(i+2)-Theta(i+1))*exp(-Theta(i+1));
                        C2 = Theta(i)/(Theta(i)-Theta(i+2))*Theta(i+1)/(Theta(i+1)-Theta(i+2))*exp(-Theta(i+2));
                        A3 = Theta(i)/(Theta(i+1)-Theta(i))*Theta(i+1)/(Theta(i+2)-Theta(i))*Theta(i+2)/(Theta(i+3)-Theta(i))*exp(-Theta(i));
                        B3 = Theta(i)/(Theta(i)-Theta(i+1))*Theta(i+1)/(Theta(i+2)-Theta(i+1))*Theta(i+2)/(Theta(i+3)-Theta(i+1))*exp(-Theta(i+1));
                        C3 = Theta(i)/(Theta(i)-Theta(i+2))*Theta(i+1)/(Theta(i+1)-Theta(i+2))*Theta(i+2)/(Theta(i+3)-Theta(i+2))*exp(-Theta(i+2));
                        D3 = Theta(i)/(Theta(i)-Theta(i+3))*Theta(i+1)/(Theta(i+1)-Theta(i+3))*Theta(i+2)/(Theta(i+2)-Theta(i+3))*exp(-Theta(i+3));
                        Pij = 1-(A0+A1+B1+A2+B2+C2+A3+B3+C3+D3);
                end
            case 4
                switch j-i
                    case 0
                        Pij = exp(-Theta(i));
                    case 1
                        A = Theta(i)/(Theta(i+1)-Theta(i))*exp(-Theta(i));
                        B = Theta(i)/(Theta(i)-Theta(i+1))*exp(-Theta(i+1));
                        Pij = A+B;
                    case 2
                        A = Theta(i)/(Theta(i+1)-Theta(i))*Theta(i+1)/(Theta(i+2)-Theta(i))*exp(-Theta(i));
                        B = Theta(i)/(Theta(i)-Theta(i+1))*Theta(i+1)/(Theta(i+2)-Theta(i+1))*exp(-Theta(i+1));
                        C = Theta(i)/(Theta(i)-Theta(i+2))*Theta(i+1)/(Theta(i+1)-Theta(i+2))*exp(-Theta(i+2));
                        Pij = A+B+C;
                    case 3
                        A0 = exp(-Theta(i));
                        A1 = Theta(i)/(Theta(i+1)-Theta(i))*exp(-Theta(i));
                        B1 = Theta(i)/(Theta(i)-Theta(i+1))*exp(-Theta(i+1));
                        A2 = Theta(i)/(Theta(i+1)-Theta(i))*Theta(i+1)/(Theta(i+2)-Theta(i))*exp(-Theta(i));
                        B2 = Theta(i)/(Theta(i)-Theta(i+1))*Theta(i+1)/(Theta(i+2)-Theta(i+1))*exp(-Theta(i+1));
                        C2 = Theta(i)/(Theta(i)-Theta(i+2))*Theta(i+1)/(Theta(i+1)-Theta(i+2))*exp(-Theta(i+2));
                        Pij = 1-(A0+A1+B1+A2+B2+C2);
                end
            case 5
                switch j-i
                    case 0
                        Pij = exp(-Theta(i));
                    case 1
                        A = Theta(i)/(Theta(i+1)-Theta(i))*exp(-Theta(i));
                        B = Theta(i)/(Theta(i)-Theta(i+1))*exp(-Theta(i+1));
                        Pij = A+B;
                    case 2
                        A0 = exp(-Theta(i));
                        A1 = Theta(i)/(Theta(i+1)-Theta(i))*exp(-Theta(i));
                        B1 = Theta(i)/(Theta(i)-Theta(i+1))*exp(-Theta(i+1));
                        Pij = 1-(A0+A1+B1);
                end
            case 6
                switch j-i
                    case 0
                        Pij = exp(-Theta(i));
                    case 1
                        A0 = exp(-Theta(i));
                        Pij = 1-(A0);
                end
            case 7
                Pij = 1;
        end
        
        %% first deravative
        fun_name = ['MUSTEM_P',num2str(i),num2str(j)];
        Pij_fun = matlabFunction(Pij,'File',fun_name,'Vars',Theta);
        Loss_fun = log(Pij);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_1'];
        f_prime_1 = matlabFunction(diff(Loss_fun,Theta1),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_2'];
        f_prime_2 = matlabFunction(diff(Loss_fun,Theta2),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_3'];
        f_prime_3 = matlabFunction(diff(Loss_fun,Theta3),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_4'];
        f_prime_4 = matlabFunction(diff(Loss_fun,Theta4),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_5'];
        f_prime_5 = matlabFunction(diff(Loss_fun,Theta5),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_6'];
        f_prime_6 = matlabFunction(diff(Loss_fun,Theta6),'File',fun_name,'Vars',Theta);
        
        
        %% second deravative
         fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_1_prime_1'];
        f_prime_1_prime_1 = matlabFunction(diff(diff(Loss_fun,Theta1)*Theta1,Theta1),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_1_prime_2'];
        f_prime_1_prime_2 = matlabFunction(diff(diff(Loss_fun,Theta1)*Theta1,Theta2),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_1_prime_3'];
        f_prime_1_prime_3 = matlabFunction(diff(diff(Loss_fun,Theta1)*Theta1,Theta3),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_1_prime_4'];
        f_prime_1_prime_4 = matlabFunction(diff(diff(Loss_fun,Theta1)*Theta1,Theta4),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_1_prime_5'];
        f_prime_1_prime_5 = matlabFunction(diff(diff(Loss_fun,Theta1)*Theta1,Theta5),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_1_prime_6'];
        f_prime_1_prime_6 = matlabFunction(diff(diff(Loss_fun,Theta1)*Theta1,Theta6),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_2_prime_1'];
        f_prime_2_prime_1 = matlabFunction(diff(diff(Loss_fun,Theta2)*Theta2,Theta1),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_2_prime_2'];
        f_prime_2_prime_2 = matlabFunction(diff(diff(Loss_fun,Theta2)*Theta2,Theta2),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_2_prime_3'];
        f_prime_2_prime_3 = matlabFunction(diff(diff(Loss_fun,Theta2)*Theta2,Theta3),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_2_prime_4'];
        f_prime_2_prime_4 = matlabFunction(diff(diff(Loss_fun,Theta2)*Theta2,Theta4),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_2_prime_5'];
        f_prime_2_prime_5 = matlabFunction(diff(diff(Loss_fun,Theta2)*Theta2,Theta5),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_2_prime_6'];
        f_prime_2_prime_6 = matlabFunction(diff(diff(Loss_fun,Theta2)*Theta2,Theta6),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_3_prime_1'];
        f_prime_3_prime_1 = matlabFunction(diff(diff(Loss_fun,Theta3)*Theta3,Theta1),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_3_prime_2'];
        f_prime_3_prime_2 = matlabFunction(diff(diff(Loss_fun,Theta3)*Theta3,Theta2),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_3_prime_3'];
        f_prime_3_prime_3 = matlabFunction(diff(diff(Loss_fun,Theta3)*Theta3,Theta3),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_3_prime_4'];
        f_prime_3_prime_4 = matlabFunction(diff(diff(Loss_fun,Theta3)*Theta3,Theta4),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_3_prime_5'];
        f_prime_3_prime_5 = matlabFunction(diff(diff(Loss_fun,Theta3)*Theta3,Theta5),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_3_prime_6'];
        f_prime_3_prime_6 = matlabFunction(diff(diff(Loss_fun,Theta3)*Theta3,Theta6),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_4_prime_1'];
        f_prime_4_prime_1 = matlabFunction(diff(diff(Loss_fun,Theta4)*Theta4,Theta1),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_4_prime_2'];
        f_prime_4_prime_2 = matlabFunction(diff(diff(Loss_fun,Theta4)*Theta4,Theta2),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_4_prime_3'];
        f_prime_4_prime_3 = matlabFunction(diff(diff(Loss_fun,Theta4)*Theta4,Theta3),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_4_prime_4'];
        f_prime_4_prime_4 = matlabFunction(diff(diff(Loss_fun,Theta4)*Theta4,Theta4),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_4_prime_5'];
        f_prime_4_prime_5 = matlabFunction(diff(diff(Loss_fun,Theta4)*Theta4,Theta5),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_4_prime_6'];
        f_prime_4_prime_6 = matlabFunction(diff(diff(Loss_fun,Theta4)*Theta4,Theta6),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_5_prime_1'];
        f_prime_5_prime_1 = matlabFunction(diff(diff(Loss_fun,Theta5)*Theta5,Theta1),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_5_prime_2'];
        f_prime_5_prime_2 = matlabFunction(diff(diff(Loss_fun,Theta5)*Theta5,Theta2),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_5_prime_3'];
        f_prime_5_prime_3 = matlabFunction(diff(diff(Loss_fun,Theta5)*Theta5,Theta3),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_5_prime_4'];
        f_prime_5_prime_4 = matlabFunction(diff(diff(Loss_fun,Theta5)*Theta5,Theta4),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_5_prime_5'];
        f_prime_5_prime_5 = matlabFunction(diff(diff(Loss_fun,Theta5)*Theta5,Theta5),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_5_prime_6'];
        f_prime_5_prime_6 = matlabFunction(diff(diff(Loss_fun,Theta5)*Theta5,Theta6),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_6_prime_1'];
        f_prime_6_prime_1 = matlabFunction(diff(diff(Loss_fun,Theta6)*Theta6,Theta1),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_6_prime_2'];
        f_prime_6_prime_2 = matlabFunction(diff(diff(Loss_fun,Theta6)*Theta6,Theta2),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_6_prime_3'];
        f_prime_6_prime_3 = matlabFunction(diff(diff(Loss_fun,Theta6)*Theta6,Theta3),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_6_prime_4'];
        f_prime_6_prime_4 = matlabFunction(diff(diff(Loss_fun,Theta6)*Theta6,Theta4),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_6_prime_5'];
        f_prime_6_prime_5 = matlabFunction(diff(diff(Loss_fun,Theta6)*Theta6,Theta5),'File',fun_name,'Vars',Theta);
        fun_name = ['MUSTEM_',num2str(i),num2str(j),'_f_prime_6_prime_6'];
        f_prime_6_prime_6 = matlabFunction(diff(diff(Loss_fun,Theta6)*Theta6,Theta6),'File',fun_name,'Vars',Theta);
    end
end