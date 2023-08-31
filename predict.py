from train import LinearRegression

if __name__ == "__main__":
    try:
        lr = LinearRegression().load()
        print("[+] Thetas loaded successfully.")
        print("[*] Thetas: [theta_0: {:.2f}, theta_1: {:.2f}]".format(lr.theta_0, lr.theta_1))
    except:
        print("[!] Error while loading thetas, did you train the model yet?")
        exit(1)
    while True:
        try:
            x = input("[#] Mileage (km) : ")
        except KeyboardInterrupt:
            break
        except EOFError:
            break
        try:
            x = float(x)
            print("[+] Estimated price: {:.2f}€".format(lr.predict(x)))
        except:
            print("[!] Invalid mileage value.")
            continue