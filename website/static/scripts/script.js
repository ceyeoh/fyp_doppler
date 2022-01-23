$(document).ready(function () {
  console.log("uploaded");
  $("form input").change(function () {
    $("form p").text(this.files[0].name + " selected");
  });
});

$(function () {
  $("#datepicker").datepicker();
});

function deleteAppointment(appointmentId) {
  fetch("/delete-appointment", {
    method: "POST",
    body: JSON.stringify({ appointmentId: appointmentId }),
  }).then((_res) => {
    window.location.href = "/appointment";
  });
}
