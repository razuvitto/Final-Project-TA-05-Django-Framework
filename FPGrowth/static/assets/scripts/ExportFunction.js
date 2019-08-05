$(document).ready(function () {
        
    $('#tableExample1').DataTable({
        "dom": 't'
    });

    $('#tableExample2').DataTable({
        "dom": "<'row'<'col-sm-6'l><'col-sm-6'f>>t<'row'<'col-sm-6'i><'col-sm-6'p>>",
        "lengthMenu": [[6, 25, 50, -1], [6, 25, 50, "All"]],
        "iDisplayLength": 6,
    });

    $('#tableExample3').DataTable({
        dom: "<'row'<'col-sm-4'l><'col-sm-4 text-center'B><'col-sm-4'f>>tp",
        "lengthMenu": [[10, 25, 50, -1], [10, 25, 50, "All"]],
        buttons: [
            {extend: 'copy', className: 'btn-sm'},
            {extend: 'csv', title: 'Frequent Pattern', className: 'btn-sm'},
            {extend: 'pdf', title: 'Frequent Pattern', className: 'btn-sm'},
            {extend: 'print', className: 'btn-sm'}
        ]
    });

    $('#tableExample4').DataTable({
        dom: "<'row'<'col-sm-4'l><'col-sm-4 text-center'B><'col-sm-4'f>>tp",
        "lengthMenu": [[10, 25, 50, -1], [10, 25, 50, "All"]],
        buttons: [
            {extend: 'copy', className: 'btn-sm'},
            {extend: 'csv', title: 'Closed Pattern', className: 'btn-sm'},
            {extend: 'pdf', title: 'Closed Pattern', className: 'btn-sm'},
            {extend: 'print', className: 'btn-sm'}
        ]
    });

    $('#tableExample5').DataTable({
        dom: "<'row'<'col-sm-4'l><'col-sm-4 text-center'B><'col-sm-4'f>>tp",
        "lengthMenu": [[10, 25, 50, -1], [10, 25, 50, "All"]],
        buttons: [
            {extend: 'copy', className: 'btn-sm'},
            {extend: 'csv', title: 'Association Rules', className: 'btn-sm'},
            {extend: 'pdf', title: 'Association Rules', className: 'btn-sm'},
            {extend: 'print', className: 'btn-sm'}
        ]
    });

});
