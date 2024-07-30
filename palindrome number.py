def palindrome (number):
    str_number = str (number)
    reverse_strnumber = str_number [::-1]

    return str_number == reverse_strnumber
num = int (input("Enter your number:") )

if palindrome(num):
    print(f"TADA! Your number {num} is a palindrome. ")
else:
    print(f"Your number{num} is not a palindrome. ")