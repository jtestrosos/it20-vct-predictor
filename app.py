from flask import Flask, flash, redirect, render_template, request, url_for

app = Flask(__name__)
app.secret_key = "super_secret_key"  # Required for flash messages

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        username = request.form.get("username")
        if username == "admin":
            flash("Welcome back, Admin!", "success")
        else:
            flash("Invalid username. Try again!", "danger")
        return redirect(url_for("home"))
    return render_template("login.html")

if __name__ == "__main__":
    app.run(debug=True)
