function readImage(input) {
    console.log(input.files);
    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
            $('#imgcontainer')
                .attr('src', e.target.result)
                .width(750)
                .height(500);
        };

        reader.readAsDataURL(input.files[0]);
    }
}