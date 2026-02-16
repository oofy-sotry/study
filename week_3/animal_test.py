class Animal:
    def speak(self):
        print("소리")

class Dog(Animal):
    def speak(self):
        print("멍멍")

if __name__ == "__main__":
    d = Dog()
    d.speak()
