(function () {
  const DICT = {
    en: {
      brand: "AgriGuard AI System",
      nav_home: "Home",
      nav_history: "History",
      nav_map: "Outbreak Map",
      app_title: "AgriGuard AI: Precision Crop Diagnostic System",
      app_subtitle: "Smart Deep Learning System for Instant Plant Disease Prediction",
    },
    hi: {
      brand: "एग्रीगार्ड एआई सिस्टम",
      nav_home: "होम",
      nav_history: "इतिहास",
      nav_map: "आउटब्रेक मैप",
      app_title: "एग्रीगार्ड एआई: सटीक फसल रोग पहचान प्रणाली",
      app_subtitle: "तुरंत पौध रोग पहचान के लिए स्मार्ट डीप लर्निंग सिस्टम",
    },
    mr: {
      brand: "एग्रीगार्ड एआय सिस्टम",
      nav_home: "मुख्यपृष्ठ",
      nav_history: "इतिहास",
      nav_map: "आऊटब्रेक मॅप",
      app_title: "एग्रीगार्ड एआय: अचूक पीಕ ರೋಗ ಪತ್ತೆ ವ್ಯವಸ್ಥೆ",
      app_subtitle: "तत्काळ रोग भाकितासाठी स्मार्ट ಡಿಪ ಲರ್ನಿಂಗ್ ಸಿಸ್ಟಮ್",
    },
    kn: {
      brand: "AgriGuard AI ಸಿಸ್ಟಮ್",
      nav_home: "ಮುಖಪುಟ",
      nav_history: "ಇತಿಹಾಸ",
      nav_map: "ರೋಗ ಹರಡುವಿಕೆ ನಕ್ಷೆ",
      app_title: "AgriGuard AI: ನಿಖರ ಬೆಳೆ ರೋಗ ಪತ್ತೆ ವ್ಯವಸ್ಥೆ",
      app_subtitle: "ತಕ್ಷಣದ ಸಸ್ಯ ರೋಗ ಪತ್ತೆಗಾಗಿ ಸ್ಮಾರ್ಟ್ ಡೀಪ್ ಲರ್ನಿಂಗ್ ವ್ಯವಸ್ಥೆ",
    },
  };

  function getLang() {
    const stored = localStorage.getItem("lang");
    if (stored && DICT[stored]) return stored;
    return "en";
  }

  function t(key) {
    const lang = getLang();
    return (DICT[lang] && DICT[lang][key]) || (DICT.en[key] || key);
  }

  function applyTranslations() {
    document.querySelectorAll("[data-i18n]").forEach((el) => {
      const key = el.getAttribute("data-i18n");
      if (!key) return;
      el.textContent = t(key);
    });

    document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
      const key = el.getAttribute("data-i18n-placeholder");
      if (!key) return;
      el.setAttribute("placeholder", t(key));
    });
  }

  window.I18N = {
    DICT,
    getLang,
    setLang: function (lang) {
      if (!DICT[lang]) lang = "en";
      localStorage.setItem("lang", lang);
      applyTranslations();
    },
    t,
    applyTranslations,
  };
})();
