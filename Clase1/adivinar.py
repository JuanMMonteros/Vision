import random as ran 

def adivinar(intentos):
    numero = ran.randint(0,100)
    numint=0
    while intentos > numint:
        entrada = input('Ingrese un entero: ')
        try:
            candidato = int(entrada) 
        except ValueError:
            print(f"¡ADVERTENCIA! '{entrada}' no es un entero válido")
            continue
        numint += 1
        if candidato == numero:
            print('Felicitaciones, adivinaste en el intento ',numint)
            break
        elif candidato < numero:
            print('No, es un poco mayor')
        else:
            print('No, es un poco menor')
    else:
        print("Se agotaron los intentos")
    
adivinar(10)
