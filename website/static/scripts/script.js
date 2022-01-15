$(document).ready(function () {
  console.log('uploaded')
  $('form input').change(function () {
    $('form p').text(this.files[0].name + " selected");
  });
});

$(function () {
  $('#datepicker').datepicker();
});