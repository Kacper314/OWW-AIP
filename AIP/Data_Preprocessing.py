import numpy as np
import pandas as pd

penguins = pd.read_csv('/home/konrad/Desktop/AiP/Project2/penguins_size.csv')
#print(penguins.head())
#print(penguins)


def ZamianaNaLiczby_Plec(Zbior_Klas):
    species = []
    for i in range(len(Zbior_Klas)):
        if Zbior_Klas[i] == "MALE":
            species.append(0)
        elif Zbior_Klas[i] == "FEMALE":
            species.append(1)
        else:
            species.append(-1)

    return species

def ZamianaNaLiczby_Wyspa(Zbior_Klas):
    species = []
    #Dream, Torgersen, or Biscoe
    for i in range(len(Zbior_Klas)):
        if Zbior_Klas[i] == "Dream":
            species.append(0)
        elif Zbior_Klas[i] == "Torgersen":
            species.append(1)
        elif Zbior_Klas[i] == "Biscoe":
            species.append(2)
        else:
            species.append(-1)

    return species

def Normalizacja(Zbior_Klas):
    normalized = []
    max = 0;
    min = 999999;
    #Dream, Torgersen, or Biscoe
    for i in range(len(Zbior_Klas)):
        if Zbior_Klas[i] > max:
            max = Zbior_Klas[i]
        if Zbior_Klas[i] < min:
            min = Zbior_Klas[i]
    for i in range(len(Zbior_Klas)):
        normalized.append((Zbior_Klas[i] - min)/(max-min))

    return normalized




train, test = np.split(penguins.sample(frac=1, random_state=17), [int(.8*len(penguins))])
train_plec = train.iloc[:, 6].values
test_plec = test.iloc[:, 6].values

train_wyspa = train.iloc[:, 1].values
test_wyspa = test.iloc[:, 1].values

train_culmen_lenght = train.iloc[:, 2].values
test_culmen_lenght = test.iloc[:, 2].values

train_waga = train.iloc[:, 5].values
test_waga = test.iloc[:, 5].values


train_flipper_lenght = train.iloc[:, 4].values
test_flipper_lenght = test.iloc[:, 4].values

train_plec_Final = ZamianaNaLiczby_Plec(train_plec)
test_plec_Final = ZamianaNaLiczby_Plec(test_plec)

train_wyspa_Final = ZamianaNaLiczby_Wyspa(train_wyspa)
test_wyspa_Final = ZamianaNaLiczby_Wyspa(test_wyspa)

train_waga_Final = Normalizacja(train_waga)
test_waga_Final = Normalizacja(test_waga)

train_flipper_lenght_Final = Normalizacja(train_plec_Final)
test_flipper_lenght_Final = Normalizacja(test_plec_Final)

train_culmen_lenght_Final = Normalizacja(train_culmen_lenght)
test_culmen_lenght_Final = Normalizacja(test_culmen_lenght)

##### Podmiana danych
train['sex'] = train_plec_Final
test['sex'] = test_plec_Final

train['island'] = train_wyspa_Final
test['island'] = test_wyspa_Final

train['body_mass_g'] = train_waga_Final
test['body_mass_g'] = test_waga_Final

train['flipper_length_mm'] = train_flipper_lenght_Final
test['flipper_length_mm'] = test_flipper_lenght_Final

train['culmen_length_mm'] = train_culmen_lenght_Final
test['culmen_length_mm'] = test_culmen_lenght_Final

print(test)





#train.to_csv('penguins_train.data', index = False, header = False)
#validation.to_csv('penguins_valid.data', index = False, header = False)
#test.to_csv('penguins_test.data', index = False, header = False)






