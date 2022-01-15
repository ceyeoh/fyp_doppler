from flask import Blueprint, render_template, request, flash

auth = Blueprint(
    "auth",
    __name__,
)

# @auth.route("/login", methods=["GET", "POST"])
@auth.route("/",  methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get('email')
        password = request.form.get('password')
        print(password)
        if len(email) < 10:
            flash("Email error", category="error")
        else:
            flash("Login successfully", category="succcess")
    return render_template("login.html")
