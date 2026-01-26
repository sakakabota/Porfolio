/* -------------------------------------------

Name: Treto
Version: 1.0
Developer: MillerDigitalDesign
Author:	bslthemes (https://bslthemes.com)

------------------------------------------- */

$(function () {

    "use strict";

    $(window).on("scroll", function () {
        var scroll = $(window).scrollTop();

        if (scroll >= 30) {
            $(".mil-top-panel").addClass("mil-active");
        } else {
            $(".mil-top-panel").removeClass("mil-active");
        }
    });

    var swiper = new Swiper('.mil-timeline-slider', {
        slidesPerView: 1,
        spaceBetween: 30,
        speed: 800,
        parallax: true,
        navigation: {
            prevEl: '.mil-timeline-prev',
            nextEl: '.mil-timeline-next',
        },
        pagination: {
            el: '.mil-timeline-pagination',
            type: 'fraction',
            clickable: true,
        },
        breakpoints: {
            992: {
                slidesPerView: 2,
            },
        },
    });

    var swiper = new Swiper('.mil-timeline-slider-2', {
        slidesPerView: 1,
        spaceBetween: 30,
        speed: 800,
        parallax: true,
        navigation: {
            prevEl: '.mil-timeline-2-prev',
            nextEl: '.mil-timeline-2-next',
        },
        pagination: {
            el: '.mil-timeline-2-pagination',
            type: 'fraction',
            clickable: true,
        },
        breakpoints: {
            992: {
                slidesPerView: 3,
            },
        },
    });

    var swiper = new Swiper('.mil-reviews-slider', {
        slidesPerView: 1,
        spaceBetween: 30,
        speed: 800,
        parallax: true,
        navigation: {
            prevEl: '.mil-reviews-prev',
            nextEl: '.mil-reviews-next',
        },
        pagination: {
            el: '.mil-reviews-pagination',
            type: 'fraction',
            clickable: true,
        },
    });

    $('.mil-menu-btn').on('click', function () {
        $('.mil-menu-btn , .mil-top-panel nav').toggleClass('mil-active');
    });

    $('.mil-filter a').on('click', function () {
        $('.mil-filter .mil-current').removeClass('mil-current');
        $(this).addClass('mil-current');

        var selector = $(this).data('filter');
        $('.mil-portfolio-grid').isotope({
            filter: selector
        });
        return false;

    });

    var $container = $('.mil-portfolio-grid').isotope({
        itemSelector: '.mil-grid-item',
        transitionDuration: '0.5s',
        masonry: {
            columnWidth: '.grid-sizer'
        }
    });

    /**
		Image Popup
	**/
	$('.mfp-image').magnificPopup();

    /*
		Gallery popup
	*/
	$('.mfp-gallery').on('click', function() {
		var gallery = $(this).attr('href');

		$(gallery).magnificPopup({
			delegate: 'a',
			type:'image',
			closeOnContentClick: false,
			mainClass: 'mfp-fade',
			removalDelay: 160,
			fixedContentPos: false,
			gallery: {
				enabled: true
			}
		}).magnificPopup('open');

		return false;
	});

    /**
		Validate Form
	**/
	if($('.cform').length) {
		$('#cform').validate({
			rules: {
				name: {
					required: true
				},
				tel: {
					required: true
				},
				email: {
					required: true,
					email: true
				},
                subject: {
					required: true
				},
				message: {
					required: true
				},
                checkmark: {
					required: true
				}
			},
			success: 'valid',
			submitHandler: function() {
				$.ajax({
					url: 'mailer/feedback.php',
					type: 'post',
					dataType: 'json',
					data: 'name='+ $("#cform").find('input[name="name"]').val() + '&email='+ $("#cform").find('input[name="email"]').val() + '&tel='+ $("#cform").find('input[name="tel"]').val() + '&subject='+ $("#cform").find('input[name="subject"]').val() + '&message='+ $("#cform").find('textarea[name="message"]').val(),
					beforeSend: function() {
	
					},
					complete: function() {
	
					},
					success: function(data) {
						$('#cform').fadeOut();
						$('.alert-success').delay(1000).fadeIn();
					}
				});
			}
		});
	}

    /**
		Validate Form 2
	**/
	if($('.cform-two').length) {
		$('#cform-two').validate({
			rules: {
				name: {
					required: true
				},
				email: {
					required: true,
					email: true
				},
				message: {
					required: true
				},
                checkmark: {
					required: true
				}
			},
			success: 'valid',
			submitHandler: function() {
				$.ajax({
					url: 'mailer/feedback-two.php',
					type: 'post',
					dataType: 'json',
					data: 'name='+ $("#cform-two").find('input[name="name"]').val() + '&email='+ $("#cform-two").find('input[name="email"]').val() + '&message='+ $("#cform-two").find('textarea[name="message"]').val(),
					beforeSend: function() {
	
					},
					complete: function() {
	
					},
					success: function(data) {
						$('#cform-two').fadeOut();
						$('.alert-success').delay(1000).fadeIn();
					}
				});
			}
		});
	}

});
